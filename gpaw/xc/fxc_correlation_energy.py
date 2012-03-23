import sys
from time import ctime
import numpy as np
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw import GPAW
from gpaw.xc import XC
from gpaw.response.df import DF
from gpaw.utilities import devnull
from gpaw.utilities.blas import gemmdot
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import rank, size, world
from gpaw.response.parallel import parallel_partition, set_communicator
from gpaw.sphere.lebedev import weight_n, R_nv
from scipy.special.orthogonal import p_roots

class ALDACorrelation:

    def __init__(self,
                 calc,
                 txt=None,
                 qsym=True,
                 xc=None,
                 perturbative=None,
                 lambda_points=None,
                 density_cut=None,
                 paw_correction=1,
                 unit_cell=[0,0,0]):
        
        self.calc = calc
        
        if txt is None:
            if rank == 0:
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull
        else:
            assert type(txt) is str
            from ase.parallel import paropen
            self.txt = paropen(txt, 'w')

        self.qsym = qsym
        self.nspins = calc.wfs.nspins
        self.bz_k_points = calc.wfs.bzk_kc
        self.atoms = calc.get_atoms()
        self.setups = calc.wfs.setups
        self.bz_q_points = calc.wfs.kd.get_bz_q_points()
        if qsym == False:
            self.ibz_q_points = self.bz_q_points
            self.q_weights = (np.ones(len(self.bz_q_points))
                              / len(self.bz_q_points))
        else:
            op_scc = calc.wfs.kd.symmetry.op_scc
            self.ibz_q_points = calc.wfs.kd.get_ibz_q_points(self.bz_q_points,
                                                             op_scc)[0]
            self.q_weights = calc.wfs.kd.q_weights
        
        if xc == None:
            self.xc = 'RPA'
        else:
            self.xc = xc
        
        if perturbative is not None and lambda_points is not None:
            raise 'Choose either perturbative calcalution or lambda points' 
        self.perturbative = perturbative
        self.lambda_points = lambda_points
        if self.nspins == 2 and lambda_points is None:
            self.lambda_points = 8
            self.perturbative = None
        self.density_cut = density_cut
        if self.density_cut is None:
            self.density_cut = 1.e-6
        self.paw_correction = paw_correction
        self.unit_cell = unit_cell
        self.print_initialization()
        self.initialized = 0

   
    def get_fxc_correlation_energy(self,
                                   kcommsize=1,
                                   dfcommsize=world.size,
                                   directions=None,
                                   skip_gamma=False,
                                   ecut=10,
                                   smooth_cut=None,
                                   nbands=None,
                                   gauss_legendre=None,
                                   frequency_cut=None,
                                   frequency_scale=None,
                                   w=None,
                                   restart=None):
            
        self.initialize_calculation(w,
                                    ecut,
                                    smooth_cut,
                                    nbands,
                                    kcommsize,
                                    gauss_legendre,
                                    frequency_cut,
                                    frequency_scale)

        if dfcommsize == world.size:
            self.dfcomm = world
            E_q = []
            if restart is not None:
                assert type(restart) is str
                try:
                    f = paropen(restart, 'r')
                    lines = f.readlines()
                    for line in lines:
                        E_q.append(eval(line))
                    f.close()
                    print >> self.txt, 'Correlation energy obtained ' \
                          +'from %s q-points obtained from restart file: ' \
                          % len(E_q), restart
                    print >> self.txt
                except:
                    IOError
    
            for index, q in enumerate(self.ibz_q_points[len(E_q):]):
                if abs(np.dot(q, q))**0.5 < 1.e-5:
                    E_q0 = 0.
                    if skip_gamma:
                        print >> self.txt, \
                              'Not calculating q at the Gamma point'
                        print >> self.txt
                    else:
                        if directions is None:
                            directions = [[0, 1/3.], [1, 1/3.], [2, 1/3.]]
                        for d in directions:                                   
                            E_q0 += self.E_q(q,
                                             index=index,
                                             direction=d[0]) * d[1]
                    E_q.append(E_q0)
                else:
                    E_q.append(self.E_q(q, index=index))
                    
                if restart is not None:
                    f = paropen(restart, 'a')
                    print >> f, E_q[-1]
                    f.close()
    
            E = np.dot(np.array(self.q_weights), np.array(E_q).real)

        else: # parallelzation over q points
            print >> self.txt, 'parallelization over q point ! '
            # creates q list
            qlist = []
            qweight = []
            id = 0
            for iq, q in enumerate(self.ibz_q_points):
                if abs(np.dot(q, q))**0.5 < 1.e-5:
                    if skip_gamma:
                        continue
                    else:
                        if directions is None:
                            directions = [[0, 1/3.], [1, 1/3.], [2, 1/3.]]
                        for d in directions:
                            qlist.append((id, q, d[0], d[1]))
                            qweight.append(self.q_weights[iq])
                            id += 1
                        continue
                qlist.append((id, q, 0, 1))
                qweight.append(self.q_weights[iq])
                id += 1
            nq = len(qlist)
    
            # distribute q list
            self.dfcomm, qcomm = set_communicator(world,
                                                  world.rank,
                                                  world.size,
                                                  kcommsize=dfcommsize)[:2]
            nq, nq_local, q_start, q_end = parallel_partition(nq,
                                                              qcomm.rank,
                                                              qcomm.size,
                                                              reshape=False)
    
            E_q = np.zeros(nq)
            for iq in range(q_start, q_end):
                E_q[iq] = self.E_q(qlist[iq][1],
                                   index=iq,
                                   direction=qlist[iq][2]) * qlist[iq][3]
            qcomm.sum(E_q)
    
            print >> self.txt, '(q, direction, weight), E_q, qweight'
            for iq in range(nq):
                print >> self.txt, qlist[iq][1:4], E_q[iq], qweight[iq]
    
            E = np.dot(np.array(qweight), np.array(E_q))

        print >> self.txt, '%s correlation energy:' % self.xc
        print >> self.txt, 'E_c = %s eV' % E
        print >> self.txt
        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt
        return E


    def get_E_q(self,
                kcommsize=1,
                index=None,
                q=[0., 0., 0.],
                direction=0,
                integrated=True,
                ecut=10,
                smooth_cut=None,
                nbands=None,
                gauss_legendre=None,
                frequency_cut=None,
                frequency_scale=None,
                w=None):

        self.initialize_calculation(w, ecut, smooth_cut,
                                    nbands, kcommsize,
                                    gauss_legendre, frequency_cut,
                                    frequency_scale)
        self.dfcomm = world
        E_q = self.E_q(q,
                       direction=direction,
                       integrated=integrated)
        
        print >> self.txt, 'Calculation completed at:  ', ctime()
        print >> self.txt
        print >> self.txt, \
              '------------------------------------------------------'

        return E_q


    def E_q(self,
            q,
            index=None,
            direction=0,
            integrated=True):

        if abs(np.dot(q, q))**0.5 < 1.e-5:
            q = [0.,0.,0.]
            q[direction] = 1.e-5
            optical_limit = True
        else:
            optical_limit = False

        dummy = DF(calc=self.calc,
                   xc='RPA',
                   eta=0.0,
                   w=self.w * 1j,
                   q=q,
                   ecut=self.ecut,
                   G_plus_q=True,
                   optical_limit=optical_limit,
                   hilbert_trans=False)
        dummy.txt = devnull
        dummy.initialize(simple_version=True)
        npw = dummy.npw
        del dummy
        ns = self.nspins
        
        if self.nbands is None:
            nbands = npw
        elif type(self.nbands) is float:
            nbands = int(npw * self.nbands)
        else:
            nbands = self.nbands

        if self.txt is sys.stdout:
            txt = 'response.txt'
        else:
            txt='response_'+self.txt.name
        df = DF(calc=self.calc,
                xc='RPA',
                nbands=nbands,
                eta=0.0,
                q=q,
                txt=txt,
                w=self.w * 1j,
                ecut=self.ecut,
                smooth_cut=self.smooth_cut,
                G_plus_q=True,
                density_cut=self.density_cut,
                kcommsize=self.kcommsize,
                comm=self.dfcomm,
                optical_limit=optical_limit,
                hilbert_trans=False)
        
        df.initialize(spin=0)
        Nw_local = df.Nw_local
        chi0 = np.zeros((Nw_local, npw*ns, npw*ns),
                        dtype=complex)

        if index is None:
            print >> self.txt, 'Calculating KS response function at:'
        else:
            print >> self.txt, '#', index, \
                  '- Calculating KS response function at:'
        if optical_limit:
            print >> self.txt, 'q = [0 0 0] -', 'Polarization: ', direction
        else:
            print >> self.txt, 'q = [%1.6f %1.6f %1.6f] -' \
                  % (q[0],q[1],q[2]), '%s planewaves' % npw

        df.calculate(spin=0)
        chi0[:, :npw, :npw] = df.chi0_wGG[:] 
        if ns == 2:
            print >> self.txt, 'Finished spin 0'
            del df.e_kn
            df.ecut *= Hartree
            df.xc = 'RPA'
            df.initialize(spin=1)
            df.calculate(spin=1)
            print >> self.txt, 'Finished spin 1'
            chi0[:, npw:2*npw, npw:2*npw] = df.chi0_wGG[:]
        del df.chi0_wGG

        if self.xc[:4] == 'ALDA':
            Kxc_sGG = self.calculate_Kxc(df.gd,
                                         self.calc.density.nt_sG,
                                         npw,
                                         df.Gvec_Gc,
                                         df.nG,
                                         df.vol,
                                         df.bcell_cv,
                                         self.atoms.positions/Bohr,
                                         self.calc.wfs.setups,
                                         self.calc.density.D_asp)
        elif self.xc == 'rALDA':
            Kxc_sGG, Kcr_sGG = self.calculate_ralda(df.gd,
                                                    self.calc.density.nt_sG,
                                                    npw,
                                                    df.Gvec_Gc,
                                                    df.nG,
                                                    df.vol,
                                                    df.acell_cv,
                                                    df.bcell_cv,
                                                    self.atoms.positions/Bohr,
                                                    self.calc.wfs.setups,
                                                    self.calc.density.D_asp,
                                                    q)
        elif self.xc == 'RPA':
            Kxc_sGG = np.zeros((npw, npw))
        else:
            raise 'Kernel not recognized'
        #print Kxc_sGG.real
                
        Kc_GG = np.zeros((npw, npw), dtype=complex)
        for iG in range(npw):
            qG = np.dot(df.q_c + df.Gvec_Gc[iG], df.bcell_cv)
            Kc_GG[iG,iG] = 4 * np.pi / np.dot(qG, qG)
        Kc_sGsG = np.tile(Kc_GG, (ns, ns))

        fxc_sGsG = np.zeros((npw*ns, npw*ns), dtype=complex)
        for s in range(ns):
            fxc_sGsG[s*npw:(s+1)*npw, s*npw:(s+1)*npw] = Kxc_sGG[s]

        if self.xc == 'rALDA':
            Kcr_sGsG = np.zeros((npw*ns, npw*ns), dtype=complex)
            for s in range(ns):
                Kcr_sGsG[s*npw:(s+1)*npw, s*npw:(s+1)*npw] = Kcr_sGG[s]
            if ns == 2:
                Kcr_sGsG[:npw, npw:2*npw] = Kcr_sGG[2]
                Kcr_sGsG[npw:2*npw, :npw] = Kcr_sGG[2]
            fhxc_sGsG = Kcr_sGsG + fxc_sGsG
        else:
            fhxc_sGsG = Kc_sGsG + fxc_sGsG

        #fhxc_sGsG = Kc_sGsG + fxc_sGsG
        #if self.xc == 'rALDA':
        #    fhxc_sGsG = fxc_sGsG

        local_E_q_w = np.zeros(Nw_local, dtype=complex)
        E_q_w = np.empty(len(self.w), complex)
        local_singular_w = np.zeros(Nw_local, int)
        singular_w = np.empty(len(self.w), int)
        local_negative_w = np.zeros(Nw_local, int)
        negative_w = np.empty(len(self.w), int)

        # Coupling constant integration
        if self.lambda_points is not None:
            print >> self.txt, 'Calculating interacting response function'
            ls, l_ws = p_roots(self.lambda_points)
            ls = (ls + 1.0) * 0.5
            l_ws *= 0.5            
            for i in range(Nw_local):
                chi0_fhxc = np.dot(chi0[i], fhxc_sGsG)
                for l, l_w in zip(ls, l_ws):
                    #if l == ls[-1]:
                    #    print self.w[i], np.linalg.cond(np.eye(nG*ns, nG*ns)
                    #                                    - l * chi0_fhxc).real
                    try:
                        chi_l = np.linalg.solve(np.eye(npw*ns, npw*ns)
                                                - l*chi0_fhxc, chi0[i]).real
                    except:
                        local_E_q_w[i] = np.nan
                        break
                    for s1 in range(ns):
                        for s2 in range(ns):
                            X_ss = chi_l[s1*npw:(s1+1)*npw, s2*npw:(s2+1)*npw]
                            local_E_q_w[i] -= np.trace(np.dot(X_ss, Kc_GG))*l_w
                local_E_q_w[i] += np.dot(np.diag(chi0[i]), np.diag(Kc_sGsG))
            df.wcomm.all_gather(local_E_q_w, E_q_w)
            
        # Perturbative evaluation
        elif self.perturbative is not None:
            for i in range(Nw_local):
                chi0_v = np.dot(chi0[i], Kc_sGsG)
                chi0_fhxc = np.dot(chi0[i], fhxc_sGsG)
                eigenvalues, P = np.linalg.eig(chi0_fhxc)
                if (eigenvalues.real > 1.).any():
                    local_negative_w[i] = 1
                series = np.array([1/(j+2.) * eigenvalues**(j+1)
                                   for j in range(self.perturbative)])
                A = np.dot(np.linalg.inv(P), np.dot(chi0_v, P))
                local_E_q_w[i] = -np.dot(np.diag(A), np.sum(series, 0))
            df.wcomm.all_gather(local_negative_w, negative_w)
            df.wcomm.all_gather(local_E_q_w, E_q_w)
            for i, wi in enumerate(negative_w):
                if wi:
                    print >> self.txt, \
                          'NEGATIVE EIGENVALUES AT w=%3.2f eV ' \
                          '- Use density cutoff in f_xc' % df.w_w[i].imag
        # Analytic evaluation
        else:
            for i in range(Nw_local):
                chi0_v = np.dot(chi0[i], Kc_sGsG)
                chi0_fhxc = np.dot(chi0[i], fhxc_sGsG)
                eigenvalues, P = np.linalg.eig(np.eye(npw, npw)
                                               - chi0_fhxc)
                #if i == 0:
                    #print eigenvalues.real
                    #print fhxc_sGsG.real
                    #print chi0_fhxc.real
                if (eigenvalues.real < 0).any():
                    local_negative_w[i] = 1
                if (abs(np.ones(npw*ns) - eigenvalues) < 1.e-10).any():
                    local_singular_w[i] = 1
                A = np.linalg.solve(chi0_fhxc, P)
                B = np.dot(chi0_v, A)
                C = np.dot(np.linalg.inv(P), B)
                #print np.diag(C).real
                local_E_q_w[i] = np.dot(np.diag(C), np.log(eigenvalues)) \
                                 + np.trace(chi0_v)
            df.wcomm.all_gather(local_negative_w, negative_w)
            df.wcomm.all_gather(local_singular_w, singular_w)
            df.wcomm.all_gather(local_E_q_w, E_q_w)
            for i, wi in enumerate(negative_w):
                if wi:
                    print >> self.txt, \
                          'NEGATIVE EIGENVALUES AT w=%3.2f eV ' \
                          '- Use density cutoff in f_xc' % df.w_w[i].imag
            for i, wi in enumerate(singular_w):
                if wi:
                    print >> self.txt, \
                          'SINGULAR MATRIX AT w=%3.2f eV ' \
                          '- Result may be unreliable' \
                          % df.w_w[i].imag
        del df
        if self.gauss_legendre is not None:
            E_q = np.sum(E_q_w * self.gauss_weights * self.transform) \
                  / (4*np.pi)
        else:   
            dws = self.w[1:] - self.w[:-1]
            E_q = np.dot((E_q_w[:-1] + E_q_w[1:])/2., dws) / (2.*np.pi)

        print >> self.txt, 'E_c(q) = %s eV' % E_q.real
        print >> self.txt

        if integrated:
            return E_q.real
        else:
            return E_q_w.real               


    def calculate_Kxc(self,
                      gd,
                      nt_sG,
                      npw,
                      Gvec_Gc,
                      nG,
                      vol,
                      bcell_cv,
                      R_av,
                      setups,
                      D_asp):
        
        #assert self.paw_correction in [0,1,2] # 0: No paw, 1: All paw, 2: average paw

        if self.paw_correction == 0:
            print >> self.txt, 'Calculating %s kernel - No paw correction' % self.xc
        elif self.paw_correction == 1:
            print >> self.txt, 'Calculating %s kernel - Full paw correction' % self.xc
        else:
            print >> self.txt, 'Calculating %s kernel - Average paw correction' % self.xc

        ns = self.nspins
        
        # The soft part
        if self.xc == 'ALDA_X':
            A_x = -(3/4.) * (3/np.pi)**(1/3.)
            fxc_sg = ns * (4 / 9.) * A_x * (ns*nt_sG)**(-2/3.)
        
        elif self.xc == 'ALDA':
            assert ns == 1
            fxc_sg = np.zeros_like(nt_sG)
            xc = XC(self.xc[1:])
            xc.calculate_fxc(gd, nt_sG, fxc_sg)

        else:
            raise '%s kernel not recognized' % self.xc
        
        if self.density_cut is not None:
            fxc_sg[np.where(ns*nt_sG < self.density_cut)] = 0.0        

        # FFT fxc(r)
        nG0 = nG[0] * nG[1] * nG[2]
        tmp_sg = [np.fft.fftn(fxc_sg[s]) * vol / nG0 for s in range(ns)]

        r_vg = gd.get_grid_point_coordinates()
        Kxc_sGG = np.zeros((len(fxc_sg), npw, npw), dtype=complex)
        for s in range(ns):
            for iG in range(npw):
                for jG in range(npw):
                    dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                    if (nG / 2 - np.abs(dG_c) > 0).all():
                        index = dG_c % nG
                        Kxc_sGG[s, iG, jG] = tmp_sg[s][index[0], index[1], index[2]]
                    else: # not in the fft index
                        dG_v = np.dot(dG_c, bcell_cv)
                        dGr_g = gemmdot(dG_v, r_vg, beta=0.0) 
                        Kxc_sGG[s, iG, jG] = gd.integrate(np.exp(-1j*dGr_g)*fxc_sg[s])

        if self.paw_correction == 0:
            return Kxc_sGG / vol

        # The PAW part
        KxcPAW_sGG = np.zeros_like(Kxc_sGG)
        dG_GGv = np.zeros((npw, npw, 3))
        for iG in range(npw):
            for jG in range(npw):
                dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                dG_GGv[iG, jG] =  np.dot(dG_c, bcell_cv)

        for a, setup in enumerate(setups):
            if rank == a % size:
                rgd = setup.xc_correction.rgd
                n_qg = setup.xc_correction.n_qg
                nt_qg = setup.xc_correction.nt_qg
                nc_g = setup.xc_correction.nc_g
                nct_g = setup.xc_correction.nct_g
                Y_nL = setup.xc_correction.Y_nL
                dv_g = rgd.dv_g

                D_sp = D_asp[a]
                B_pqL = setup.xc_correction.B_pqL
                D_sLq = np.inner(D_sp, B_pqL.T)

                f_sg = rgd.empty(ns)
                ft_sg = rgd.empty(ns)

                n_sLg = np.dot(D_sLq, n_qg)
                nt_sLg = np.dot(D_sLq, nt_qg)

                # Add core density
                n_sLg[:, 0] += (4 * np.pi)**0.5 / ns * nc_g
                nt_sLg[:, 0] += (4 * np.pi)**0.5 / ns * nct_g
                
                coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
                w = weight_n    

                if self.paw_correction == 2:
                    Y_nL = [Y_nL[0]]
                    w = [1.]

                for n, Y_L in enumerate(Y_nL):
                    f_sg[:] = 0.0
                    n_sg = np.dot(Y_L, n_sLg)
                    if self.xc == 'ALDA_X':
                        f_sg = ns * (4 / 9.) * A_x * (ns*n_sg)**(-2/3.)
                    else:
                        xc.calculate_fxc(rgd, n_sg, f_sg)
                    if self.density_cut is not None:
                        f_sg[np.where(ns*n_sg < self.density_cut)] = 0.0        

                    ft_sg[:] = 0.0
                    nt_sg = np.dot(Y_L, nt_sLg)
                    if self.xc == 'ALDA_X':
                        ft_sg = ns * (4 / 9.) * A_x * (ns*nt_sg)**(-2/3.)
                    else:
                        xc.calculate_fxc(rgd, nt_sg, ft_sg)
                    if self.density_cut is not None:
                        ft_sg[np.where(ns*nt_sg < self.density_cut)] = 0.0        
 
                    for i in range(len(rgd.r_g)):
                        coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n]) * rgd.r_g[i])
                        for s in range(len(f_sg)):
                            KxcPAW_sGG[s] += w[n] * np.dot(coef_GG,
                                                           (f_sg[s,i]-ft_sg[s,i]) * dv_g[i]) \
                                                           * coefatoms_GG

        world.sum(KxcPAW_sGG)
        Kxc_sGG += KxcPAW_sGG

        return Kxc_sGG / vol


    def calculate_ralda(self,
                        gd,
                        nt_sG,
                        npw,
                        Gvec_Gc,
                        nG,
                        vol,
                        acell_cv,
                        bcell_cv,
                        R_av,
                        setups,
                        D_asp,
                        q):

        ns = self.nspins

        Kxc_sGG = np.zeros((ns, npw, npw), dtype=complex)
        Kcr_sGG = np.zeros((ns+3%ns, npw, npw), dtype=complex)

        #assert self.paw_correction in [0,1,2] # 0: No paw, 1: All paw, 2: average paw
        if self.paw_correction == 0:
            print >> self.txt, 'Calculating %s kernel - No paw correction' % self.xc
        elif self.paw_correction == 1:
            print >> self.txt, 'Calculating %s kernel - Paw correction at ALDA level' % self.xc
        else:
            print >> self.txt, 'Calculating %s kernel - Average paw correction' % self.xc
            
        A_x = -(3/4.) * (3/np.pi)**(1/3.)
        if rank == 0:
            # The soft part        
            fx_sg = ns * (4 / 9.) * A_x * (ns*nt_sG)**(-2/3.)

            flocal_sg = 4 * ns * nt_sG * fx_sg
            Vlocal_sg = 4 * (3 * ns * nt_sG/ np.pi)**(1./3.)
            if ns == 2:
                Vlocaloff_g = 4 * (3 * (nt_sG[0] + nt_sG[1])/ np.pi)**(1./3.)

            nG0 = nG[0] * nG[1] * nG[2]
            r_vg = gd.get_grid_point_coordinates()
            q_v = np.dot(q, bcell_cv) 
            ls, l_ws = p_roots(12)

            # Unit cells
            R = []
            R_weight = []
            N_R = self.unit_cell
            N_k = self.calc.wfs.kd.N_c
            assert (np.array(N_R) < np.array(N_k)).all()
            N_k0 = float(N_k[0]*N_k[1]*N_k[2])
            for i in range(-N_R[0], N_R[0]+1):
                for j in range(-N_R[1], N_R[1]+1):
                    for h in range(-N_R[2], N_R[2]+1):
                        R.append(i*acell_cv[0] + j*acell_cv[1] + h*acell_cv[2])
                        R_weight.append((N_k[0]-abs(i))*(N_k[1]-abs(j))*(N_k[2]-abs(h)) / N_k0)

            Kxc_sGr = np.zeros((ns, npw,
                                np.shape(r_vg[0])[0],
                                np.shape(r_vg[0])[1],
                                np.shape(r_vg[0])[2]), dtype=complex)
            Kcr_sGr = np.zeros((ns+3%ns, npw,
                                np.shape(r_vg[0])[0],
                                np.shape(r_vg[0])[1],
                                np.shape(r_vg[0])[2]), dtype=complex)

            for s in range(ns):
                if ns == 2:
                    print >> self.txt, '    Spin:', s
                # Loop of Lattice points
                for i, R_i in enumerate(R):
                    print >> self.txt, '    Lattice point and weight %s:' % i, (R_i*Bohr), R_weight[i]
                    # Loop over r'. f_rr is a function of r represented by the full array r_vg
                    for g_x in range(nG[0]):
                        if (np.array(N_R) == 0).all():
                            print >> self.txt, '        Grid point', g_x, 'out of', nG[0]
                        for g_y in range(nG[1]):
                            for g_z in range(nG[2]):
                                 r_x = r_vg[0, g_x, g_y, g_z] + R_i[0]
                                 r_y = r_vg[1, g_x, g_y, g_z] + R_i[1]
                                 r_z = r_vg[2, g_x, g_y, g_z] + R_i[2]
                                 #r_p = r_vg[:, g_x, g_y, g_z] + R_i
                                 rr = ((r_vg[0]-r_x)**2 +
                                       (r_vg[1]-r_y)**2 +
                                       (r_vg[2]-r_z)**2)**0.5 # |r-r'-R_i|                        

                                 # Renormalized f_x term
                                 n_av = ns*(nt_sG[s] + nt_sG[s, g_x, g_y, g_z]) / 2.
                                 k_f = (3 * np.pi**2 * n_av)**(1./3.)
                                 x = 2 * k_f * rr
                                 fx_g = ns * (4 / 9.) * A_x * n_av**(-2/3.)
                                 f_rr = fx_g / (2*np.pi**2*rr**3) * (np.sin(x) - x*np.cos(x))

                                 # Renormalized Hartree Term
                                 y = np.array([(l + 1.)*x / 2. for l in ls])
                                 y_w = np.array([l_w * x / 2. for l_w in l_ws])
                                 V_rr = np.sum(np.sin(y)/y * y_w, 0) * 2 / np.pi / rr

                                 # Off diagonal Hartree term
                                 if s == 1:
                                     n_spin = (nt_sG[0] + nt_sG[1] +
                                               nt_sG[0, g_x, g_y, g_z] + nt_sG[1, g_x, g_y, g_z]) / 2.
                                     k_f = (3 * np.pi**2 * n_spin)**(1./3.)
                                     x = 2 * k_f * rr
                                     y = np.array([(l + 1.)*x / 2. for l in ls])
                                     y_w = np.array([l_w * x / 2. for l_w in l_ws])
                                     V_off = np.sum(np.sin(y)/y * y_w, 0) * 2 / np.pi / rr

                                 # Terms with r = r'
                                 if (np.abs(R_i) < 0.001).all():
                                     f_rr[g_x, g_y, g_z] = flocal_sg[s, g_x, g_y, g_z]
                                     V_rr[g_x, g_y, g_z] = Vlocal_sg[s, g_x, g_y, g_z]
                                     if s == 1:
                                         V_off[g_x, g_y, g_z] = Vlocaloff_g[g_x, g_y, g_z]

                                 f_rr[np.where(n_av < self.density_cut)] = 0.0        
                                 V_rr[np.where(n_av < self.density_cut)] = 0.0
                                 if s == 1:
                                     V_off[np.where(n_av < self.density_cut)] = 0.0

                                 f_rr *= R_weight[i]        
                                 V_rr *= R_weight[i]
                                 if s == 1:
                                     V_off *= R_weight[i]

                                 # Fourier transform of r
                                 r_r = np.array([r_vg[0]-r_x, r_vg[1]-r_y, r_vg[2]-r_z]) # r-r'
                                 e_q = np.exp(-1j * gemmdot(q_v, r_r, beta=0.0))
                                 tmp_Kxc = np.fft.fftn(f_rr*e_q) * vol / nG0
                                 tmp_Kcr = np.fft.fftn(V_rr*e_q) * vol / nG0
                                 if s == 1:
                                     tmp_Koff = np.fft.fftn(V_off*e_q) * vol / nG0
                                 for iG in range(npw): 
                                     if (nG / 2 - np.abs(Gvec_Gc[iG]) > 0).all():
                                         index = Gvec_Gc[iG] % nG
                                         Kxc_sGr[s, iG, g_x, g_y, g_z] += tmp_Kxc[index[0], index[1], index[2]]
                                         Kcr_sGr[s, iG, g_x, g_y, g_z] += tmp_Kcr[index[0], index[1], index[2]]
                                         if s == 1:
                                             Kcr_sGr[2, iG, g_x, g_y, g_z] += tmp_Koff[index[0], index[1], index[2]]
                                     else: # not in the fft index
                                         G_v = np.dot(Gvec_Gc[iG], bcell_cv)
                                         Gr_g = gemmdot(G_v, r_vg, beta=0.0)
                                         Kxc_sGr[s, iG, g_x, g_y, g_z] += gd.integrate(np.exp(-1j*Gr_g)*e_q*f_rr)
                                         Kcr_sGr[s, iG, g_x, g_y, g_z] += gd.integrate(np.exp(-1j*Gr_g)*e_q*V_rr)
                                         if s == 1:
                                             Kcr_sGr[2, iG, g_x, g_y, g_z] += gd.integrate(np.exp(-1j*Gr_g)*e_q*V_off)

                # Fourier transform of r'
                for iG in range(npw):
                    tmp_Kxc = np.fft.fftn(Kxc_sGr[s,iG]) * vol / nG0
                    tmp_Kcr = np.fft.fftn(Kcr_sGr[s,iG]) * vol / nG0
                    if s == 1:
                        tmp_Koff = np.fft.fftn(Kcr_sGr[2,iG]) * vol / nG0
                    for jG in range(npw):
                        if (nG / 2 - np.abs(Gvec_Gc[jG]) > 0).all():
                            index = -Gvec_Gc[jG] % nG
                            Kxc_sGG[s, iG, jG] = tmp_Kxc[index[0], index[1], index[2]]
                            Kcr_sGG[s, iG, jG] = tmp_Kcr[index[0], index[1], index[2]]
                            if s == 1:
                                Kcr_sGG[2, iG, jG] += tmp_Koff[index[0], index[1], index[2]]
                        else: # not in the fft index
                            G_v = np.dot(Gvec_Gc[jG], bcell_cv)
                            Gr_g = gemmdot(G_v, r_vg, beta=0.0)
                            Kxc_sGG[s, iG, jG] = gd.integrate(np.exp(1j*Gr_g)*Kxc_sGr[s, iG])
                            Kcr_sGG[s, iG, jG] = gd.integrate(np.exp(1j*Gr_g)*Kcr_sGr[s, iG])
                            if s == 1:
                                Kcr_sGG[2, iG, jG] = gd.integrate(np.exp(1j*Gr_g)*Kcr_sGr[2, iG])
            del Kxc_sGr
            del Kcr_sGr
        world.sum(Kxc_sGG)
        world.sum(Kcr_sGG)

        if self.paw_correction == 0:
            return Kxc_sGG / vol, Kcr_sGG / vol

        # The PAW part
        print >> self.txt, '    Calculating PAW corrections'
        KxcPAW_sGG = np.zeros_like(Kxc_sGG)
        dG_GGv = np.zeros((npw, npw, 3))
        for iG in range(npw):
            for jG in range(npw):
                dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                dG_GGv[iG, jG] =  np.dot(dG_c, bcell_cv)

        for a, setup in enumerate(setups):
            if rank == a % size:
                rgd = setup.xc_correction.rgd
                n_qg = setup.xc_correction.n_qg
                nt_qg = setup.xc_correction.nt_qg
                nc_g = setup.xc_correction.nc_g
                nct_g = setup.xc_correction.nct_g
                Y_nL = setup.xc_correction.Y_nL
                dv_g = rgd.dv_g

                D_sp = D_asp[a]
                B_pqL = setup.xc_correction.B_pqL
                D_sLq = np.inner(D_sp, B_pqL.T)

                f_sg = rgd.empty(ns)
                ft_sg = rgd.empty(ns)

                n_sLg = np.dot(D_sLq, n_qg)
                nt_sLg = np.dot(D_sLq, nt_qg)

                # Add core density
                n_sLg[:, 0] += (4 * np.pi)**0.5 / ns * nc_g
                nt_sLg[:, 0] += (4 * np.pi)**0.5 / ns * nct_g
                
                coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
                w = weight_n    

                if self.paw_correction == 2:
                    Y_nL = [Y_nL[0]]
                    w = [1.]

                for n, Y_L in enumerate(Y_nL):
                    f_sg[:] = 0.0
                    n_sg = np.dot(Y_L, n_sLg)
                    f_sg = ns * (4 / 9.) * A_x * (ns*n_sg)**(-2/3.)
                    if self.density_cut is not None:
                        f_sg[np.where(ns*n_sg < self.density_cut)] = 0.0        

                    ft_sg[:] = 0.0
                    nt_sg = np.dot(Y_L, nt_sLg)
                    ft_sg = ns * (4 / 9.) * A_x * (ns*nt_sg)**(-2/3.)
                    if self.density_cut is not None:
                        ft_sg[np.where(ns*nt_sg < self.density_cut)] = 0.0        
 
                    for i in range(len(rgd.r_g)):
                        coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n]) * rgd.r_g[i])
                        for s in range(len(f_sg)):
                            KxcPAW_sGG[s] += w[n] * np.dot(coef_GG,
                                                           (f_sg[s,i]-ft_sg[s,i]) * dv_g[i]) \
                                                           * coefatoms_GG

        world.sum(KxcPAW_sGG)
        Kxc_sGG += KxcPAW_sGG

        return Kxc_sGG / vol, Kcr_sGG / vol
                
    def initialize_calculation(self, w, ecut, smooth_cut,
                               nbands, kcommsize,
                               gauss_legendre, frequency_cut, frequency_scale):
        if w is not None:
            assert (gauss_legendre is None and
                    frequency_cut is None and
                    frequency_scale is None)
        else:
            if gauss_legendre is None:
                gauss_legendre = 16
            self.gauss_points, self.gauss_weights = p_roots(gauss_legendre)
            if frequency_scale is None:
                frequency_scale = 2.0
            if frequency_cut is None:
                frequency_cut = 800.
            ys = 0.5 - 0.5 * self.gauss_points
            ys = ys[::-1]
            w = (-np.log(1-ys))**frequency_scale
            w *= frequency_cut/w[-1]
            alpha = (-np.log(1-ys[-1]))**frequency_scale/frequency_cut
            transform = (-np.log(1-ys))**(frequency_scale-1) \
                        / (1-ys)*frequency_scale/alpha
            self.transform = transform
            
        dummy = DF(calc=self.calc,
                   xc='RPA',
                   eta=0.0,
                   w=w * 1j,
                   q=[0.,0.,0.0001],
                   ecut=ecut,
                   optical_limit=True,
                   hilbert_trans=False,
                   kcommsize=kcommsize)
        dummy.txt = devnull
        dummy.initialize(simple_version=True)

        self.ecut = ecut
        self.smooth_cut = smooth_cut
        self.w = w
        self.gauss_legendre = gauss_legendre
        self.frequency_cut = frequency_cut
        self.frequency_scale = frequency_scale
        self.kcommsize = kcommsize
        self.nbands = nbands

        if self.perturbative is None and self.lambda_points is None:
            print >> self.txt, 'Analytical coupling constant integration'
        if self.perturbative is not None:
            print >> self.txt, 'Perturbative evaluation of Dyson equation' \
                  + ' to order %s in f_hxc' % self.perturbative
        if self.lambda_points is not None:
            print >> self.txt, 'Numerical coupling constant integration' \
                  + ' with % s Gauss-Legendre points' % self.lambda_points
        print >> self.txt
        print >> self.txt, 'Planewave cutoff              : %s eV' % ecut
        if self.smooth_cut is not None:
            print >> self.txt, 'Smooth cutoff from            : %s x cutoff' \
                  % self.smooth_cut
        print >> self.txt, 'Number of Planewaves at Gamma : %s' % dummy.npw
        if self.nbands is None:
            print >> self.txt, 'Response function bands       :' \
                  + ' Equal to number of Planewaves'
        elif type(self.nbands) is float:
            print >> self.txt, 'Response function bands       : %s' \
                  % int(dummy.npw * self.nbands)
        else:
            print >> self.txt, 'Response function bands       : %s' \
                  % self.nbands
        if self.density_cut is not None:
            print >> self.txt
            print >> self.txt, 'Min value of pseudo density   : %1.2e Bohr^-3' \
                  % np.min(self.calc.density.nt_sG)
            print >> self.txt, 'Max value of pseudo density   : %1.2e Bohr^-3' \
                  % np.max(self.calc.density.nt_sG)
            print >> self.txt, 'Density cutoff in fxc at      : %1.2e Bohr^-3' \
                  % self.density_cut
        print >> self.txt, 'Frequencies'
        if self.gauss_legendre is not None:
            print >> self.txt, '    Gauss-Legendre integration '\
                  + 'with %s frequency points' % len(self.w)
            print >> self.txt, '    Frequency cutoff is '\
                  + '%s eV and scale (B) is %s' % (self.w[-1],
                                                  self.frequency_scale)
        else:
            print >> self.txt, '    %s specified frequency points' \
                  % len(self.w)
            print >> self.txt, '    Frequency cutoff is %s eV' \
                  % self.w[-1]
        print >> self.txt
        print >> self.txt, 'Parallelization scheme'
        print >> self.txt, '     Total CPUs        : %d' % dummy.comm.size
        if dummy.nkpt == 1:
            print >> self.txt, '     Band parsize      : %d' % dummy.kcomm.size
        else:
            print >> self.txt, '     Kpoint parsize    : %d' % dummy.kcomm.size
        print >> self.txt, '     Frequency parsize : %d' % dummy.wScomm.size
        print >> self.txt, 'Memory usage estimate'
        print >> self.txt, '     chi0_wGG(Q)       : %f M / cpu' \
              % (dummy.Nw_local * dummy.npw**2 * 16. / 1024**2)
        print >> self.txt
        del dummy


    def print_initialization(self):
        
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, 'Non-self-consistent %s correlation energy' \
              % self.xc
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, 'Started at:  ', ctime()
        print >> self.txt
        print >> self.txt, 'Atoms                          :   %s' \
              % self.atoms.get_chemical_formula(mode="hill")
        print >> self.txt, 'Ground state XC functional     :   %s' \
              % self.calc.hamiltonian.xc.name
        print >> self.txt, 'Valence electrons              :   %s' \
              % self.setups.nvalence
        print >> self.txt, 'Number of Bands                :   %s' \
              % self.calc.wfs.bd.nbands
        print >> self.txt, 'Number of Converged Bands      :   %s' \
              % self.calc.input_parameters['convergence']['bands']
        print >> self.txt, 'Number of Spins                :   %s' \
              % self.nspins
        print >> self.txt, 'Number of k-points             :   %s' \
              % len(self.calc.wfs.bzk_kc)
        print >> self.txt, 'Number of q-points             :   %s' \
              % len(self.bz_q_points)
        print >> self.txt, 'Number of Irreducible k-points :   %s' \
              % len(self.calc.wfs.ibzk_kc)
        if self.qsym:
            print >> self.txt, 'Number of Irreducible q-points :   %s' \
                  % len(self.ibz_q_points)
        else:
            print >> self.txt, 'No reduction of q-points' 
        print >> self.txt
        for q, weight in zip(self.ibz_q_points, self.q_weights):
            print >> self.txt, 'q: [%1.4f %1.4f %1.4f] - weight: %1.3f' \
                  % (q[0],q[1],q[2], weight)
        print >> self.txt
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt
