import sys
from time import ctime
import numpy as np
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw import GPAW
from gpaw.xc import XC
from gpaw.xc.libxc import LibXC
from gpaw.response.df import DF
from gpaw.utilities import devnull
from gpaw.utilities.blas import gemmdot, axpy
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import rank, size, world, serial_comm
from gpaw.blacs import BlacsGrid, BlacsDescriptor, Redistributor
from gpaw.response.parallel import parallel_partition, set_communicator
from gpaw.fd_operators import Gradient, Laplace
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.io.tar import Writer, Reader
from scipy.special import sici
from scipy.special.orthogonal import p_roots

class FXCCorrelation:

    def __init__(self,
                 calc,
                 txt=None,
                 qsym=True,
                 xc=None,
                 num=0,
                 write=False,
                 method='standard',
                 lambda_points=8,
                 density_cut=None,
                 paw_correction=1,
                 unit_cells=None):
        
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
        self.num = num
        self.nspins = calc.wfs.nspins
        self.bz_k_points = calc.wfs.bzk_kc
        self.atoms = calc.get_atoms()
        self.setups = calc.wfs.setups
        self.bz_q_points = calc.wfs.kd.get_bz_q_points(first=True)
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

        self.lambda_points = lambda_points
        assert method in ['solid', 'standard']
        self.method = method             
        self.density_cut = density_cut
        if self.density_cut is None:
            self.density_cut = 1.e-6
        assert paw_correction in range(4)
        self.paw_correction = paw_correction
        self.unit_cells = unit_cells
        self.print_initialization()
        self.initialized = 0
        self.grad_v = [Gradient(calc.density.gd, v, n=1).apply for v in range(3)]
        self.laplace = Laplace(calc.density.gd, n=1).apply
        self.write = write
   
    def get_fxc_correlation_energy(self,
                                   kcommsize=1,
                                   serial_w=False,
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
                                    serial_w,
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
                            if serial_w:
                                E_q0 += self.E_q_serial_w(q,
                                                          index=index,
                                                          direction=d[0]) * d[1]
                            else:
                                E_q0 += self.E_q(q,
                                                 index=index,
                                                 direction=d[0]) * d[1]
                    E_q.append(E_q0)
                else:
                    if serial_w:
                        E_q.append(self.E_q_serial_w(q, index=index))
                    else:
                        E_q.append(self.E_q(q, index=index))
                    
                if restart is not None:
                    f = paropen(restart, 'a')
                    print >> f, E_q[-1]
                    f.close()
    
            E = np.dot(np.array(self.q_weights), np.array(E_q).real)

        else: # parallelzation over q points
            print >> self.txt, 'parallelization over q point ! '
            assert serial_w is False
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
                serial_w=None,
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
                                    nbands, kcommsize, serial_w,
                                    gauss_legendre, frequency_cut,
                                    frequency_scale)
        self.dfcomm = world
        if serial_w:
            E_q = self.E_q_serial_w(q,
                                    direction=direction,
                                    integrated=integrated)
        else:
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

        fhxc_sGsG, Kc_G = self.get_fxc(q,
                                       optical_limit,
                                       index,
                                       direction)

        ns = self.nspins

        print >> self.txt, 'Calculating KS response function'

        if self.nbands is None:
            nbands = len(Kc_G)
        elif type(self.nbands) is float:
            nbands = int(len(Kc_G) * self.nbands)
        else:
            nbands = self.nbands

        if self.txt is sys.stdout:
            txt = 'response.txt'
        else:
            txt='response_' + self.txt.name
        df = DF(calc=self.calc,
                xc=None,
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
        
        df.initialize()
        Nw_local = df.Nw_local
        npw = df.npw
        Gvec_Gc = df.Gvec_Gc
        chi0 = np.zeros((Nw_local, ns*npw, ns*npw), dtype=complex)

        df.calculate(seperate_spin=0)
        chi0[:, :npw, :npw] = df.chi0_wGG[:] 
        if ns == 2:
            df.ecut *= Hartree
            df.xc = 'RPA'
            df.initialize()
            df.calculate(seperate_spin=1)
            chi0[:, npw:2*npw, npw:2*npw] = df.chi0_wGG[:]
        del df.chi0_wGG
        
        #w = Writer('chi0_w')
        
        local_E_q_w = np.zeros(Nw_local, dtype=complex)
        E_q_w = np.empty(len(self.w), complex)

        print >> self.txt, 'Calculating interacting response function'
        ls, l_ws = p_roots(self.lambda_points)
        ls = (ls + 1.0) * 0.5
        l_ws *= 0.5            
        for i in range(Nw_local):
            chi0_fhxc = np.dot(chi0[i], fhxc_sGsG)
            for l, l_w in zip(ls, l_ws):
                chi_l = np.linalg.solve(np.eye(npw*ns, npw*ns)
                                        - l*chi0_fhxc, chi0[i]).real
                for s1 in range(ns):
                    for s2 in range(ns):
                        X_ss = chi_l[s1*npw:(s1+1)*npw, s2*npw:(s2+1)*npw]
                        local_E_q_w[i] -= np.dot(np.diag(X_ss), Kc_G)*l_w
            local_E_q_w[i] += np.dot(np.diag(chi0[i]), np.tile(Kc_G, ns))
        df.wcomm.all_gather(local_E_q_w, E_q_w)

        del df, chi0, chi0_fhxc, chi_l, X_ss, Kc_G, fhxc_sGsG
        
        if self.gauss_legendre is not None:
            E_q = np.sum(E_q_w * self.gauss_weights * self.transform) \
                  / (4 * np.pi)
        else:   
            dws = self.w[1:] - self.w[:-1]
            E_q = np.dot((E_q_w[:-1] + E_q_w[1:])/2., dws) / (2 * np.pi)

        print >> self.txt, 'E_c(q) = %s eV' % E_q.real
        print >> self.txt

        if integrated:
            return E_q.real
        else:
            return E_q_w.real               


    def get_fxc(self, q, optical_limit, index, direction):

        dummy = DF(calc=self.calc,
                   eta=0.0,
                   w=self.w * 1j,
                   q=q,
                   optical_limit=optical_limit,
                   ecut=self.ecut,
                   G_plus_q=True,
                   hilbert_trans=False)
        dummy.txt = devnull
        dummy.initialize(simple_version=True)
        gd = dummy.gd
        npw = dummy.npw
        Gvec_Gc = dummy.Gvec_Gc
        nG = dummy.nG
        vol = dummy.vol
        bcell_cv = dummy.bcell_cv
        acell_cv = dummy.acell_cv
        del dummy       

        if index is not None:
            print >> self.txt, '#', index
        if optical_limit:
            print >> self.txt, 'q = [0 0 0] -', 'Polarization: ', direction
        else: print >> self.txt, 'q = [%1.6f %1.6f %1.6f] -' \
              % (q[0],q[1],q[2]), '%s planewaves' % npw

        print >> self.txt, 'Calculating %s kernel' % self.xc

        Kc_G = np.zeros(npw, dtype=complex)
        for iG in range(npw):
            qG = np.dot(q + Gvec_Gc[iG], bcell_cv)
            Kc_G[iG] = 4 * np.pi / np.dot(qG, qG)

        if self.xc == 'RPA':
            Kc_GG = np.zeros((npw, npw), dtype=complex)
            for iG in range(npw):
                Kc_GG[iG, iG] = Kc_G[iG]
            fhxc_sGsG = np.tile(Kc_GG, (self.nspins, self.nspins))

        else:
            if self.xc[0] == 'r':
                if self.method == 'solid':
                    fhxc_sGsG = self.calculate_rkernel_solid(gd,
                                                             npw,
                                                             Gvec_Gc,
                                                             nG,
                                                             vol,
                                                             acell_cv,
                                                             bcell_cv,
                                                             q)
                elif self.method == 'standard':
                    fhxc_sGsG = self.calculate_rkernel(gd,
                                                       npw,
                                                       Gvec_Gc,
                                                       nG,
                                                       vol,
                                                       acell_cv,
                                                       bcell_cv,
                                                       q)
                else:
                    raise 'Method % s not known' % self.method
            else:
                fhxc_sGsG = self.calculate_local_kernel(gd,
                                                        npw,
                                                        Gvec_Gc,
                                                        nG,
                                                        vol,
                                                        bcell_cv)
                Kc_GG = np.zeros((npw, npw), dtype=complex)
                for iG in range(npw):
                    Kc_GG[iG, iG] = Kc_G[iG]
                fhxc_sGsG += np.tile(Kc_GG, (self.nspins, self.nspins))

        return fhxc_sGsG, Kc_G

    def calculate_rkernel(self,
                          gd,
                          npw,
                          Gvec_Gc,
                          nG,
                          vol,
                          acell_cv,
                          bcell_cv,
                          q):

        ns = self.nspins

        if self.paw_correction == 1:
            nt_sG = np.array([self.calc.get_all_electron_density(gridrefinement=1, spin=s)
                              for s in range(ns)]) * Bohr**3 * (ns % 2 +1)
        else:
            nt_sG = self.calc.density.nt_sG

        fhxc_sGsG = np.zeros((ns*npw, ns*npw), dtype=complex)

        if self.xc[2:] == 'LDA':
            A_x = -(3/4.) * (3/np.pi)**(1/3.)
            fx_sg = ns * (4 / 9.) * A_x * (ns * nt_sG)**(-2/3.)
        else:
            nt_sg = np.array([self.calc.get_all_electron_density(gridrefinement=2, spin=s)
                              for s in range(ns)]) * Bohr**3 * (ns % 2 +1)
            fx_sg = ns * np.array([self.get_fxc_g(ns * nt_sg[s])
                                   for s in range(ns)])
            
        flocal_sg = 4 * ns * nt_sG * fx_sg
        Vlocal_sg = 4 * (3 * ns * nt_sG / np.pi)**(1./3.)
        if ns == 2:
            Vlocaloff_g = 4 * (3 * (nt_sG[0] + nt_sG[1])/ np.pi)**(1./3.)

        nG0 = nG[0] * nG[1] * nG[2]
        r_vg = gd.get_grid_point_coordinates()
        r_vgx = r_vg[0].flatten()
        r_vgy = r_vg[1].flatten()
        r_vgz = r_vg[2].flatten()
        
        q_v = np.dot(q, bcell_cv) 

        # Unit cells
        R = []
        R_weight = []
        if self.unit_cells is None:
            N_R = self.calc.wfs.kd.N_c
        else:
            N_R = self.unit_cells
        N_R0 = N_R[0]*N_R[1]*N_R[2]
        for i in range(-N_R[0]+1, N_R[0]):
            for j in range(-N_R[1]+1, N_R[1]):
                for h in range(-N_R[2]+1, N_R[2]):
                    R.append(i*acell_cv[0] + j*acell_cv[1] + h*acell_cv[2])
                    R_weight.append((N_R[0]-abs(i))*
                                    (N_R[1]-abs(j))*
                                    (N_R[2]-abs(h)) / float(N_R0))
        if N_R0 > 1:
            print >> self.txt, '    Lattice point sampling: ' \
                  + '(%s x %s x %s)^2 ' % (N_R[0], N_R[1], N_R[2]) \
                  + ' Reduced to %s lattice points' % len(R)
        
        l_g_size = -(-nG0 // world.size)
        l_g_range = range(world.rank * l_g_size,
                          min((world.rank+1) * l_g_size, nG0))

        fhxc_sGr = np.zeros((ns+3%ns, npw, len(l_g_range)), dtype=complex)

        inv_error = np.seterr()['invalid']
        np.seterr(invalid='ignore')

        for s in range(ns):
            if ns == 2:
                print >> self.txt, '    Spin:', s
            # Loop over Lattice points
            for i, R_i in enumerate(R):
                # Loop over r'.
                # f_rr, V_rr and V_off are functions of r (dim. as r_vg[0])
                for g in l_g_range:
                    #print g
                    r_x = r_vgx[g] + R_i[0]
                    r_y = r_vgy[g] + R_i[1]
                    r_z = r_vgz[g] + R_i[2]

                    # |r-r'-R_i|
                    rr = ((r_vg[0]-r_x)**2 +
                          (r_vg[1]-r_y)**2 +
                          (r_vg[2]-r_z)**2)**0.5

                    # Renormalized f_xc term
                    n_av = ns*(nt_sG[s] + nt_sG[s].flatten()[g]) / 2.
                    k_f = (3 * np.pi**2 * n_av)**(1./3.)
                    x = 2 * k_f * rr
                    if self.xc[2:] == 'LDA':
                        fx_g = ns * (4 / 9.) * A_x * n_av**(-2/3.)
                    else:
                        n_av_g = ns*(nt_sg[s] + nt_sG[s].flatten()[g]) / 2.
                        fx_g = ns * self.get_fxc_g(n_av_g)

                    f_rr = fx_g * (np.sin(x) - x*np.cos(x)) \
                           / (2 * np.pi**2 * rr**3)

                    # Renormalized Hartree Term
                    V_rr = sici(x)[0] * 2 / np.pi / rr

                    # Off diagonal Hartree term
                    if s == 1:
                        n_spin = (nt_sG[0] + nt_sG[1] +
                                  nt_sG[0].flatten()[g] +
                                  nt_sG[1].flatten()[g]) / 2.
                        k_f = (3 * np.pi**2 * n_spin)**(1/3.)
                        x = 2 * k_f * rr
                        V_off = sici(x)[0] * 2 / np.pi / rr

                    # Terms with r = r'
                    if (np.abs(R_i) < 0.001).all():
                        tmp_flat = f_rr.flatten()
                        tmp_flat[g] = flocal_sg[s].flatten()[g]
                        f_rr = tmp_flat.reshape((nG[0], nG[1], nG[2]))
                        tmp_flat = V_rr.flatten()
                        tmp_flat[g] = Vlocal_sg[s].flatten()[g]
                        V_rr = tmp_flat.reshape((nG[0], nG[1], nG[2]))
                        if s == 1:
                            tmp_flat = V_off.flatten()
                            tmp_flat[g] = Vlocaloff_g.flatten()[g]
                            V_off = tmp_flat.reshape((nG[0], nG[1], nG[2]))
                        del tmp_flat
                        
                    f_rr[np.where(n_av < self.density_cut)] = 0.0        
                    V_rr[np.where(n_av < self.density_cut)] = 0.0
                    if s == 1:
                        V_off[np.where(n_spin < self.density_cut)] = 0.0

                    f_rr *= R_weight[i]        
                    V_rr *= R_weight[i]
                    if s == 1:
                        V_off *= R_weight[i]

                    # r-r'-R_i
                    r_r = np.array([r_vg[0]-r_x, r_vg[1]-r_y, r_vg[2]-r_z])
                    
                    # Fourier transform of r
                    e_q = np.exp(-1j * gemmdot(q_v, r_r, beta=0.0))
                    tmp_fhxc = np.fft.fftn(f_rr*e_q) * vol / nG0
                    tmp_fhxc += np.fft.fftn(V_rr*e_q) * vol / nG0
                    if s == 1:
                        tmp_V_off = np.fft.fftn(V_off*e_q) * vol / nG0
                    for iG in range(npw):
                        assert (nG / 2 - np.abs(Gvec_Gc[iG]) > 0).all()
                        f_i = Gvec_Gc[iG] % nG
                        fhxc_sGr[s, iG, g-l_g_range[0]] += \
                                    tmp_fhxc[f_i[0], f_i[1], f_i[2]]
                        if s == 1:
                            fhxc_sGr[2, iG, g-l_g_range[0]] += \
                                        tmp_V_off[f_i[0], f_i[1], f_i[2]]
                        
            l_pw_size = -(-npw // world.size)
            l_pw_range = range(world.rank * l_pw_size,
                               min((world.rank+1) * l_pw_size, npw))

            if world.size > 1 : 
                bg1 = BlacsGrid(world, 1, world.size)
                bg2 = BlacsGrid(world, world.size, 1)
                bd1 = bg1.new_descriptor(npw, nG0, npw, -(-nG0 / world.size))
                bd2 = bg2.new_descriptor(npw, nG0, -(-npw / world.size), nG0)

                fhxc_Glr = np.zeros((len(l_pw_range), nG0), dtype=complex)
                if s == 1:
                    Koff_Glr = np.zeros((len(l_pw_range), nG0), dtype=complex)

                r = Redistributor(bg1.comm, bd1, bd2) 
                r.redistribute(fhxc_sGr[s], fhxc_Glr, npw, nG0)
                if s == 1:
                    r.redistribute(fhxc_sGr[2], Koff_Glr, npw, nG0)
            else:
                fhxc_Glr = fhxc_sGr[s]
                if s == 1:
                    Koff_Glr = fhxc_sGr[2]

            # Fourier transform of r'
            for iG in range(len(l_pw_range)):
                tmp_fhxc = np.fft.fftn(fhxc_Glr[iG].reshape(nG)) * vol/nG0
                if s == 1:
                    tmp_Koff = np.fft.fftn(Koff_Glr[iG].reshape(nG)) * vol/nG0
                for jG in range(npw):
                    assert (nG / 2 - np.abs(Gvec_Gc[jG]) > 0).all()
                    f_i = -Gvec_Gc[jG] % nG
                    fhxc_sGsG[s*npw + l_pw_range[0] + iG, s*npw + jG] = \
                                    tmp_fhxc[f_i[0], f_i[1], f_i[2]]
                    if s == 1:
                        fhxc_sGsG[npw + l_pw_range[0] + iG, jG] += \
                                      tmp_Koff[f_i[0], f_i[1], f_i[2]]

        np.seterr(divide=inv_error)

        del fhxc_sGr, fhxc_Glr

        world.sum(fhxc_sGsG)

        if ns == 2:
            fhxc_sGsG[:npw, npw:] = fhxc_sGsG[npw:, :npw]

        if self.paw_correction in [0, 2]:
            print >> self.txt, '    Calculating PAW correction'
            f_paw_sGG = self.add_paw_correction(npw,
                                                Gvec_Gc,
                                                bcell_cv,
                                                self.calc.wfs.setups,
                                                self.calc.density.D_asp,
                                                self.atoms.positions/Bohr)
            for s in range(ns):
                fhxc_sGsG[s*npw:(s+1)*npw, s*npw:(s+1)*npw] += f_paw_sGG[s]
            
        return fhxc_sGsG / vol


    def calculate_rkernel_solid(self,
                                gd,
                                npw,
                                Gvec_Gc,
                                nG,
                                vol,
                                acell_cv,
                                bcell_cv,
                                q):

        ns = self.nspins

        if self.paw_correction == 1:
            nt_sG = np.array([self.calc.get_all_electron_density(gridrefinement=1, spin=s)
                              for s in range(ns)]) * Bohr**3 * (ns % 2 +1)
        else:
            nt_sG = self.calc.density.nt_sG

        fhxc_sGsG = np.zeros((ns*npw, ns*npw), dtype=complex)

        if self.num:
            print >> self.txt, 'Numerical evaluation of local kernel'
            nt_sg = np.array([self.calc.get_all_electron_density(gridrefinement=2, spin=s)
                              for s in range(ns)]) * Bohr**3 * (ns % 2 +1)
            fx_sg = self.get_numerical_fxc_sg(nt_sg)
        else:
            if self.xc[2:] == 'LDA':
                A_x = -(3/4.) * (3/np.pi)**(1/3.)
                fx_sg = ns * (4 / 9.) * A_x * (ns * nt_sG)**(-2/3.)
            else:
                nt_sg = np.array([self.calc.get_all_electron_density(gridrefinement=2, spin=s)
                                  for s in range(ns)]) * Bohr**3 * (ns % 2 +1)
                fx_sg = ns * np.array([self.get_fxc_g(ns * nt_sg[s])
                                       for s in range(ns)])

        fx_sg[np.where(nt_sG < self.density_cut)] = 0.0        

        r_vg = gd.get_grid_point_coordinates()

        fhxc_sGsG = np.zeros((ns*npw, ns*npw), dtype=complex)

        kf_s = (3 * np.pi**2 * ns * nt_sG)**(1./3.)
        #v_c_g = 4 * np.pi * np.ones(np.shape(fxc_sg[0]), float)
        if ns == 2:
            kf_off = (3 * np.pi**2 * (nt_sG[0] + nt_sG[1]))**(1./3.)
            v_off_sg = 4 * np.pi * np.ones(np.shape(fx_sg[0]), float)

        l_pw_size = -(-npw // world.size)
        l_pw_range = range(world.rank * l_pw_size,
                           min((world.rank+1) * l_pw_size, npw))
        
        for s in range(ns):
            if ns == 2:
                print >> self.txt, '    Spin: ', s
            for iG in l_pw_range:
                for jG in range(npw):
                    dGq_c = (Gvec_Gc[iG] + Gvec_Gc[jG])/ 2. + q
                    if (np.abs(dGq_c) < 1.e-12).all():
                        dGq_c = np.array([0.0, 0.0, 0.00001])
                    dGq_v = np.dot(dGq_c, bcell_cv)
                    fx = fx_sg[s].copy()
                    fx[np.where(2*kf_s[s] < np.dot(dGq_v, dGq_v)**0.5)] = 0.0
                    v_c = 4 * np.pi * np.ones(np.shape(fx_sg[0]), float)
                    v_c /= np.dot(dGq_v, dGq_v)
                    v_c[np.where(2*kf_s[s] < np.dot(dGq_v, dGq_v)**0.5)] = 0.0
                    if s == 1:
                        v_off = 4 * np.pi * np.ones(np.shape(fx_sg[0]), float)
                        v_off /= np.dot(dGq_v, dGq_v)
                        v_off[np.where(2*kf_off < np.dot(dGq_v, dGq_v)**0.5)] = 0.0
                    dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                    dG_v = np.dot(dG_c, bcell_cv)
                    dGr_g = gemmdot(dG_v, r_vg, beta=0.0)
                    fhxc_sGsG[s*npw+iG, s*npw+jG] = gd.integrate(np.exp(-1j*dGr_g) * (fx + v_c))
                    if s == 1:
                        fhxc_sGsG[iG, npw+jG] = gd.integrate(np.exp(-1j*dGr_g)
                                                             * v_off)
        if ns == 2:
            fhxc_sGsG[npw:2*npw, :npw] = fhxc_sGsG[:npw, npw:2*npw]

        world.sum(fhxc_sGsG)

        if self.paw_correction in [0, 2]:
            print >> self.txt, '    Calculating PAW correction'
            f_paw_sGG = self.add_paw_correction(npw,
                                                Gvec_Gc,
                                                bcell_cv,
                                                self.calc.wfs.setups,
                                                self.calc.density.D_asp,
                                                self.atoms.positions/Bohr)
            for s in range(ns):
                fhxc_sGsG[s*npw:(s+1)*npw, s*npw:(s+1)*npw] += f_paw_sGG[s]
            
        return fhxc_sGsG / vol
    

    def calculate_local_kernel(self,
                               gd,
                               npw,
                               Gvec_Gc,
                               nG,
                               vol,
                               bcell_cv):
        
        ns = self.nspins

        if self.paw_correction == 1:
            nt_sG = np.array([self.calc.get_all_electron_density(gridrefinement=1, spin=s)
                              for s in range(ns)]) * Bohr**3 * (ns % 2 +1)
        else:
            nt_sG = self.calc.density.nt_sG

        fhxc_sGsG = np.zeros((ns*npw, ns*npw), dtype=complex)

        A_x = -(3/4.) * (3/np.pi)**(1/3.)
        fxc_sg = ns * (4 / 9.) * A_x * (ns*nt_sG)**(-2/3.)
        fxc_sg[np.where(nt_sG < self.density_cut)] = 0.0        

        r_vg = gd.get_grid_point_coordinates()

        fhxc_sGsG = np.zeros((ns*npw, ns*npw), dtype=complex)

        l_pw_size = -(-npw // world.size)
        l_pw_range = range(world.rank * l_pw_size,
                           min((world.rank+1) * l_pw_size, npw))
        
        for s in range(ns):
            if ns == 2:
                print >> self.txt, '    Spin: ', s
            for iG in l_pw_range:
                for jG in range(npw):
                    fxc = fxc_sg[s].copy()
                    dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                    dG_v = np.dot(dG_c, bcell_cv)
                    dGr_g = gemmdot(dG_v, r_vg, beta=0.0)
                    fhxc_sGsG[s*npw+iG, s*npw+jG] = gd.integrate(np.exp(-1j*dGr_g) * fxc)

        world.sum(fhxc_sGsG)

        if self.paw_correction in [0, 2]:
            print >> self.txt, '    Calculating PAW correction'
            f_paw_sGG = self.add_paw_correction(npw,
                                                Gvec_Gc,
                                                bcell_cv,
                                                self.calc.wfs.setups,
                                                self.calc.density.D_asp,
                                                self.atoms.positions/Bohr)
            for s in range(ns):
                fhxc_sGsG[s*npw:(s+1)*npw, s*npw:(s+1)*npw] += f_paw_sGG[s]
            
        return fhxc_sGsG / vol


    def add_paw_correction(self, npw, Gvec_Gc, bcell_cv, setups, D_asp, R_av):
        ns = self.nspins
        A_x = -(3/4.) * (3/np.pi)**(1/3.)
        KxcPAW_sGG = np.zeros((ns, npw, npw), complex)
        dG_GGv = np.zeros((npw, npw, 3))
        for iG in range(npw):
            for jG in range(npw):
                dG_c = Gvec_Gc[iG] - Gvec_Gc[jG]
                dG_GGv[iG, jG] =  np.dot(dG_c, bcell_cv)

        for a, setup in enumerate(setups):
            rgd = setup.xc_correction.rgd
            ng = len(rgd.r_g)
            myng = -(-ng // world.size)
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

                for i in range(world.rank * myng,
                               min((world.rank + 1) * myng, ng)):
                    coef_GG = np.exp(-1j * np.inner(dG_GGv, R_nv[n])
                                     * rgd.r_g[i])
                    for s in range(len(f_sg)):
                        KxcPAW_sGG[s] += w[n] * np.dot(coef_GG,
                                            (f_sg[s,i]-ft_sg[s,i]) * dv_g[i]) \
                                                       * coefatoms_GG

        world.sum(KxcPAW_sGG)

        return KxcPAW_sGG
                

    def initialize_calculation(self, w, ecut, smooth_cut,
                               nbands, kcommsize, serial_w,
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
        if serial_w:
            dummy.Nw_local = 1
            dummy.wScomm.size = 1
            
        self.ecut = ecut
        self.smooth_cut = smooth_cut
        self.w = w
        self.gauss_legendre = gauss_legendre
        self.frequency_cut = frequency_cut
        self.frequency_scale = frequency_scale
        self.kcommsize = kcommsize
        self.nbands = nbands

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
            print >> self.txt, 'Min value of pseudo density   : %1.2e Bohr^-3'\
                  % np.min(self.calc.density.nt_sG)
            print >> self.txt, 'Max value of pseudo density   : %1.2e Bohr^-3'\
                  % np.max(self.calc.density.nt_sG)
            print >> self.txt, 'Density cutoff in fxc at      : %1.2e Bohr^-3'\
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
        print >> self.txt, '     chi0_wGG(q)         : %6.3f MB / frequency point' \
              % (self.nspins**2 * dummy.npw**2 * 16. / 1024**2)
        print >> self.txt, '     Kernel_GG(q)        : %6.3f MB / CPU' \
              % ((2*self.nspins + 3%self.nspins) * dummy.npw**2 * 16. / 1024**2)
        if self.method == 'standard':
            n0 = dummy.gd.N_c[0] * dummy.gd.N_c[1] * dummy.gd.N_c[2]
            print >> self.txt, '     Kernel_rG(q) (Int.) : %6.3f MB / CPU' \
                      % ((2*self.nspins + 3%self.nspins)
                         * dummy.npw * float(n0)/world.size * 16. / 1024**2)
            
        print >> self.txt
        del dummy


    def print_initialization(self):
        
        print >> self.txt, \
              '------------------------------------------------------'
        print >> self.txt, 'Non-self-consistent %s correlation energy' \
              % self.xc
        if self.xc is not 'RPA':
            if self.xc[0] == 'r':
                if self.method == 'solid':
                    print >> self.txt, 'Periodic average density'
                else:
                    print >> self.txt, 'Non-periodic two-point density'
            if self.paw_correction == 0:
                print >> self.txt, 'Using pseudo-density with ALDA PAW corrections'
            elif self.paw_correction == 1:
                print >> self.txt, 'Using all-electron density' 
            elif self.paw_correction == 2:
                print >> self.txt, 'Using pseudo-density with average ALDA PAW corrections'
            elif self.paw_correction == 3:
                print >> self.txt, 'Using pseudo-density'
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


    def get_C6_coefficient(self,
                           ecut=100.,
                           smoothcut=None,
                           nbands=None,
                           kcommsize=1,
                           extrapolate=False,
                           gauss_legendre=None,
                           frequency_cut=None,
                           frequency_scale=None,
                           direction=2):

        self.initialize_calculation(None,
                                    ecut,
                                    None,
                                    nbands,
                                    kcommsize,
                                    extrapolate,
                                    gauss_legendre,
                                    frequency_cut,
                                    frequency_scale)

        d = direction
        d_pro = []
        for i in range(3):
            if i != d:
                d_pro.append(i)
        
        dummy = DF(calc=self.calc,
                   eta=0.0,
                   w=self.w * 1j,
                   ecut=self.ecut,
                   hilbert_trans=False)
        dummy.txt = devnull
        dummy.initialize(simple_version=True)
        npw = dummy.npw
        del dummy

        q = [0.,0.,0.]
        q[d] = 1.e-5

        fhxc_sGsG, Kc_G = self.get_fxc(q, True, 0, d)

        if self.nbands is None:
            nbands = npw
        else:
            nbands = self.nbands

        if self.txt is sys.stdout:
            txt = 'response.txt'
        else:
            txt='response_'+self.txt.name
        df = DF(calc=self.calc,
                xc=None,
                nbands=nbands,
                eta=0.0,
                q=q,
                txt=txt,
                w=self.w * 1j,
                ecut=self.ecut,
                comm=world,
                optical_limit=True,
                G_plus_q=True,
                kcommsize=self.kcommsize,
                hilbert_trans=False)
        
        print >> self.txt, 'Calculating RPA response function'
        print >> self.txt, 'Polarization: %s' % d

        df.initialize()
        ns = self.nspins
        Nw_local = df.Nw_local
        chi0 = np.zeros((Nw_local, ns*npw, ns*npw), dtype=complex)

        df.calculate(seperate_spin=0)
        chi0[:, :npw, :npw] = df.chi0_wGG[:] 
        if ns == 2:
            df.ecut *= Hartree
            df.xc = 'RPA'
            df.initialize()
            df.calculate(seperate_spin=1)
            chi0[:, npw:2*npw, npw:2*npw] = df.chi0_wGG[:]
        del df.chi0_wGG

        local_a0_w = np.zeros(Nw_local, dtype=complex)
        a0_w = np.empty(len(self.w), complex)
        local_a_w = np.zeros(Nw_local, dtype=complex)
        a_w = np.empty(len(self.w), complex)

        Gvec_Gv = np.dot(df.Gvec_Gc + np.array(q), df.bcell_cv)
        gd = self.calc.density.gd
        n_d = gd.get_size_of_global_array()[d]
        d_d = gd.get_grid_spacings()[d]
        r_d = np.array([i*d_d for i in range(n_d)])

        print >> self.txt, 'Calculating real space integrals'

        int_G = np.zeros(npw, complex)
        for iG in range(npw):
            if df.Gvec_Gc[iG, d_pro[0]] == 0 and df.Gvec_Gc[iG, d_pro[1]] == 0:
                int_G[iG] = np.sum(r_d * np.exp(1j*Gvec_Gv[iG, d] * r_d))*d_d
        int2_GG = np.outer(int_G, int_G.conj())

        print >> self.txt, 'Calculating dynamic polarizability'

        for i in range(Nw_local):
            chi0_fhxc = np.dot(chi0[i], fhxc_sGsG)
            chi = np.linalg.solve(np.eye(npw*ns, npw*ns)
                                  - chi0_fhxc, chi0[i]).real
            for s1 in range(ns):
                X0 = chi0[i, s1*npw:(s1+1)*npw, s1*npw:(s1+1)*npw]
                local_a0_w[i] += np.trace(np.dot(X0, int2_GG))
                for s2 in range(ns):
                    X_ss = chi[s1*npw:(s1+1)*npw, s2*npw:(s2+1)*npw]
                    local_a_w[i] += np.trace(np.dot(X_ss, int2_GG))
        df.wcomm.all_gather(local_a0_w, a0_w)
        df.wcomm.all_gather(local_a_w, a_w)

        A = df.vol / gd.cell_cv[d,d]
        a0_w *= A**2 / df.vol
        a_w *= A**2 / df.vol

        del df, chi0, chi0_fhxc, chi, X_ss, Kc_G, fhxc_sGsG
        
        C06 = np.sum(a0_w**2 * self.gauss_weights
                     * self.transform) * 3 / (2*np.pi)
        C6 = np.sum(a_w**2 * self.gauss_weights
                    * self.transform) * 3 / (2*np.pi)

        print >> self.txt, 'C06 = %s Ha*Bohr**6' % (C06.real / Hartree)
        print >> self.txt, 'C6 = %s Ha*Bohr**6' % (C6.real / Hartree)
        print >> self.txt

        return C6.real / Hartree, C06.real / Hartree


    def get_fxc_g(self, n_g):
        
        gd = self.calc.density.gd.refine()

        xc = XC('GGA_X_' + self.xc[2:])
        #xc = XC('LDA_X')
        #sigma = np.zeros_like(n_g).flat[:]
        xc.set_grid_descriptor(gd)
        sigma_xg, gradn_svg = xc.calculate_sigma(np.array([n_g]))

        dedsigma_xg = np.zeros_like(sigma_xg)
        e_g = np.zeros_like(n_g)
        v_sg = np.array([np.zeros_like(n_g)])
        
        xc.calculate_gga(e_g, np.array([n_g]), v_sg, sigma_xg, dedsigma_xg)

        sigma = sigma_xg[0].flat[:]
        gradn_vg = gradn_svg[0]
        dedsigma_g = dedsigma_xg[0]
        
        libxc = LibXC('GGA_X_' + self.xc[2:])
        #libxc = LibXC('LDA_X')
        libxc.initialize(1)
        libxc_fxc = libxc.xc.calculate_fxc_spinpaired
        
        fxc_g = np.zeros_like(n_g).flat[:]
        d2edndsigma_g = np.zeros_like(n_g).flat[:]
        d2ed2sigma_g = np.zeros_like(n_g).flat[:]

        libxc_fxc(n_g.flat[:], fxc_g, sigma, d2edndsigma_g, d2ed2sigma_g)
        fxc_g = fxc_g.reshape(np.shape(n_g))
        d2edndsigma_g = d2edndsigma_g.reshape(np.shape(n_g))
        d2ed2sigma_g = d2ed2sigma_g.reshape(np.shape(n_g))

        tmp = np.zeros_like(fxc_g)
        tmp1 = np.zeros_like(fxc_g)
        
        for v in range(3):
            self.grad_v[v](d2edndsigma_g * gradn_vg[v], tmp)
            axpy(-4.0, tmp, fxc_g)
            
        for u in range(3):
            for v in range(3):
                self.grad_v[v](d2ed2sigma_g * gradn_vg[u] * gradn_vg[v], tmp)
                self.grad_v[u](tmp, tmp1)
                axpy(4.0, tmp1, fxc_g)

        self.laplace(dedsigma_g, tmp)
        axpy(2.0, tmp, fxc_g)
            
        return fxc_g[::2,::2,::2]


    def get_numerical_fxc_sg(self, n_sg):

        gd = self.calc.density.gd.refine()

        delta = 1.e-4
        if self.xc[2:] == 'LDA':
            xc = XC('LDA_X')
            v1xc_sg = np.zeros_like(n_sg)
            v2xc_sg = np.zeros_like(n_sg)
            xc.calculate(gd, (1+delta)*n_sg, v1xc_sg)
            xc.calculate(gd, (1-delta)*n_sg, v2xc_sg)
            fxc_sg = (v1xc_sg - v2xc_sg) / (2 * delta * n_sg)
        else:
            fxc_sg = np.zeros_like(n_sg)
            xc = XC('GGA_X_' + self.xc[2:])
            vxc_sg = np.zeros_like(n_sg)
            xc.calculate(gd, n_sg, vxc_sg)
            for s in range(len(n_sg)):
                for x in range(len(n_sg[0])):
                    for y in range(len(n_sg[0,0])):
                        for z in range(len(n_sg[0,0,0])):
                            v1xc_sg = np.zeros_like(n_sg)
                            n1_sg = n_sg.copy()
                            n1_sg[s,x,y,z] *= (1 + delta)
                            xc.calculate(gd, n1_sg, v1xc_sg)
                            fxc_sg[s,x,y,z] = (v1xc_sg[s,x,y,z] - vxc_sg[s,x,y,z]) \
                                           / (delta * n_sg[s,x,y,z])
                            
        return fxc_sg[:,::2,::2,::2]
