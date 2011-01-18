import sys
from time import time, ctime
import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw import GPAW, extra_parameters
from gpaw.utilities import unpack, devnull
from gpaw.utilities.blas import gemmdot, gemv, scal, axpy
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.fd_operators import Gradient
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.response.symmetrize import find_kq, find_ibzkpt, symmetrize_wavefunction
from gpaw.response.math_func import delta_function, hilbert_transform, \
     two_phi_planewave_integrals
from gpaw.response.parallel import set_communicator, \
     parallel_partition, SliceAlongFrequency, SliceAlongOrbitals
from gpaw.response.kernel import calculate_Kxc
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.memory import maxrss

class CHI:
    """This class is a calculator for the linear density response function.

    Parameters:

        nband: int
            Number of bands.
        wmax: float
            Maximum energy for spectrum.
        dw: float
            Frequency interval.
        wlist: tuple
            Frequency points.
        q: ndarray
            Momentum transfer in reduced coordinate.
        Ecut: ndarray
            Planewave cutoff energy.
        eta: float
            Spectrum broadening factor.
        sigma: float
            Width for delta function.
    """

    def __init__(self,
                 calc=None,
                 nbands=None,
                 w=None,
                 q=None,
                 ecut=10.,
                 eta=0.2,
                 ftol=1e-7,
                 txt=None,
                 hilbert_trans=True,
                 optical_limit=False,
                 kcommsize=None):

        self.xc = 'LDA'
        self.nspin = 1

        self.txtname = txt
        self.output_init()

        if isinstance(calc, str):
            # Always use serial_communicator when a filename is given.
            self.calc = GPAW(calc, communicator=serial_comm, txt=None)
        else:
            # To be optimized so that the communicator is loaded automatically 
            # according to kcommsize.
            # 
            # so temporarily it is used like this :
            # kcommsize = int (should <= world.size)
            # r0 = rank % kcommsize
            # ranks = np.arange(r0, r0+size, kcommsize)
            # calc = GPAW(filename.gpw, communicator=ranks, txt=None)
            self.calc = calc

        self.nbands = nbands
        self.q_c = q

        self.w_w = w
        self.eta = eta
        self.ftol = ftol
        if type(ecut) is int or type(ecut) is float:
            self.ecut = np.ones(3) * ecut
        else:
            assert len(ecut) == 3
            self.ecut = np.array(ecut, dtype=float)
        self.hilbert_trans = hilbert_trans
        self.optical_limit = optical_limit
        self.kcommsize = kcommsize
        self.comm = world
        self.chi0_wGG = None


    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------')
        self.printtxt('Response function calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        # Frequency init
        if len(self.w_w) == 1:
            self.HilberTrans = False
            
        if self.hilbert_trans:
            self.dw = self.w_w[1] - self.w_w[0]
            assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
            assert self.w_w.max() == self.w_w[-1]
            
            self.dw /= Hartree
            self.w_w  /= Hartree
            self.wmax = self.w_w[-1] 
            self.wcut = self.wmax + 5. / Hartree
            self.Nw  = int(self.wmax / self.dw) + 1
            self.NwS = int(self.wcut / self.dw) + 1
        else:
            self.Nw = len(self.w_w)
            self.NwS = 0
            self.dw = None
            
        self.eta /= Hartree
        self.ecut /= Hartree

        calc = self.calc
        
        # kpoint init
        self.bzk_kc = calc.get_bz_k_points()
        self.ibzk_kc = calc.get_ibz_k_points()
        self.nkpt = self.bzk_kc.shape[0]
        self.ftol /= self.nkpt

        # band init
        if self.nbands is None:
            self.nbands = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence

        assert calc.wfs.nspins == 1

        # cell init
        self.acell_cv = calc.atoms.cell / Bohr
        self.bcell_cv, self.vol, self.BZvol = get_primitive_cell(self.acell_cv)

        # grid init
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        gd = GridDescriptor(self.nG, calc.wfs.gd.cell_cv, pbc_c=True, comm=serial_comm)
        self.gd = gd        
        self.h_cv = gd.h_cv

        # obtain eigenvalues, occupations
        nibzkpt = self.ibzk_kc.shape[0]
        kweight_k = calc.get_k_point_weights()

        try:
            self.e_kn
        except:
            self.printtxt('Use eigenvalues from the calculator.')
            self.e_kn = np.array([calc.get_eigenvalues(kpt=k)
                    for k in range(nibzkpt)]) / Hartree
            self.printtxt('Eigenvalues(k=0) are:')
            print  >> self.txt, self.e_kn[0] * Hartree
        self.f_kn = np.array([calc.get_occupation_numbers(kpt=k) / kweight_k[k]
                    for k in range(nibzkpt)]) / self.nkpt

        if self.hilbert_trans:
            # for band parallelization.
            for n in range(self.nbands):
                if (self.f_kn[:, n] - self.ftol < 0).all():
                    self.nvalbands = n
                    break
        else:
            # if not hilbert transform, all the bands should be used.
            self.nvalbands = self.nbands

        # k + q init
        assert self.q_c is not None
        self.qq_v = np.dot(self.q_c, self.bcell_cv) # summation over c

        if self.optical_limit:
            kq_k = np.arange(self.nkpt)
            self.expqr_g = 1.
        else:
            r_vg = gd.get_grid_point_coordinates() # (3, nG)
            qr_g = gemmdot(self.qq_v, r_vg, beta=0.0)
            self.expqr_g = np.exp(-1j * qr_g)
            del r_vg, qr_g
            kq_k = find_kq(self.bzk_kc, self.q_c)
        self.kq_k = kq_k

        # Plane wave init
        self.npw, self.Gvec_Gc, self.Gindex_G = set_Gvectors(self.acell_cv, self.bcell_cv, self.nG, self.ecut)

        # Projectors init
        setups = calc.wfs.setups
        pt = LFC(gd, [setup.pt_j for setup in setups],
                 calc.wfs.kpt_comm, dtype=calc.wfs.dtype, forces=True)
        spos_ac = calc.atoms.get_scaled_positions()
        pt.set_k_points(self.bzk_kc)
        pt.set_positions(spos_ac)
        self.pt = pt

        # Symmetry operations init
        usesymm = calc.input_parameters.get('usesymm')
        if usesymm == None or self.nkpt == 1:
            op_scc = (np.eye(3, dtype=int),)
        elif usesymm == False:
            op_scc = (np.eye(3, dtype=int), -np.eye(3, dtype=int))
        else:
            op_scc = calc.wfs.symmetry.op_scc
        self.op_scc = op_scc

        # Parallelization initialize
        self.parallel_init()

        # Printing calculation information
        self.print_stuff()

        if extra_parameters.get('df_dry_run'):
            raise SystemExit

        # For LCAO wfs
        if calc.input_parameters['mode'] == 'lcao':
            calc.initialize_positions()        
        self.printtxt('     GS calculator   : %f M / cpu' %(maxrss() / 1024**2))
        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        kk_Gv = gemmdot(self.q_c + self.Gvec_Gc, self.bcell_cv.copy(), beta=0.0)
        phi_aGp = {}
        for a, id in enumerate(setups.id_a):
            phi_aGp[a] = two_phi_planewave_integrals(kk_Gv, setups[a])
            for iG in range(self.npw):
                phi_aGp[a][iG] *= np.exp(-1j * 2. * pi *
                                         np.dot(self.q_c + self.Gvec_Gc[iG], spos_ac[a]) )

        # For optical limit, G == 0 part should change
        if self.optical_limit:
            for a, id in enumerate(setups.id_a):
                nabla_iiv = setups[a].nabla_iiv
                phi_aGp[a][0] = -1j * (np.dot(nabla_iiv, self.qq_v)).ravel()

        self.phi_aGp = phi_aGp
        self.printtxt('')
        self.printtxt('Finished phi_Gp !')

        # Calculate ALDA kernel for EELS spectrum
        # Use RPA kernel for Optical spectrum
        if not self.optical_limit:
            R_av = calc.atoms.positions / Bohr
            self.Kxc_GG = calculate_Kxc(self.gd, # global grid
                                    calc.density.nt_sG,
                                    self.npw, self.Gvec_Gc,
                                    self.nG, self.vol,
                                    self.bcell_cv, R_av,
                                    calc.wfs.setups,
                                    calc.density.D_asp)

            self.printtxt('Finished ALDA kernel ! ')
        else:
            self.Kxc_GG = np.zeros((self.npw, self.npw))
            self.printtxt('Use RPA for optical spectrum ! ')
            self.printtxt('')
            
        return


    def calculate(self):
        """Calculate the non-interacting density response function. """

        calc = self.calc
        gd = self.gd
        sdisp_cd = gd.sdisp_cd
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        pt = self.pt
        f_kn = self.f_kn
        e_kn = self.e_kn

        # Matrix init
        chi0_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype=complex)
        if self.hilbert_trans:
            specfunc_wGG = np.zeros((self.NwS_local, self.npw, self.npw), dtype = complex)

        # Prepare for the derivative of pseudo-wavefunction
        if self.optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

        rho_G = np.zeros(self.npw, dtype=complex)
        t0 = time()
        t_get_wfs = 0
        for k in range(self.kstart, self.kend):

            # Find corresponding kpoint in IBZ
            ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[k])
            if self.optical_limit:
                ibzkpt2, iop2, timerev2 = ibzkpt1, iop1, timerev1
            else:
                ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[kq_k[k]])

            for n in range(self.nstart, self.nend):
#                print >> self.txt, k, n, t_get_wfs, time() - t0
                t1 = time()
                psitold_g = self.get_wavefunction(ibzkpt1, n, k, True)
                t_get_wfs += time() - t1
                psit1new_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop1], ibzk_kc[ibzkpt1],
                                                      bzk_kc[k], timerev1)
                P1_ai = pt.dict()
                pt.integrate(psit1new_g, P1_ai, k)

                psit1_g = psit1new_g.conj() * self.expqr_g

                for m in range(self.nbands):

		    if self.hilbert_trans:
			check_focc = (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol
                    else:
                        check_focc = np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol

                    t1 = time()
                    psitold_g = self.get_wavefunction(ibzkpt2, m, k, check_focc)
                    t_get_wfs += time() - t1

                    if check_focc:
                        psit2_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop2], ibzk_kc[ibzkpt2],
                                                           bzk_kc[kq_k[k]], timerev2)

                        P2_ai = pt.dict()
                        pt.integrate(psit2_g, P2_ai, kq_k[k])

                        # fft
                        tmp_g = np.fft.fftn(psit2_g*psit1_g) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex_G[iG]
                            rho_G[iG] = tmp_g[index[0], index[1], index[2]]

                        if self.optical_limit:
                            phase_cd = np.exp(2j * pi * sdisp_cd * bzk_kc[kq_k[k], :, np.newaxis])
                            for ix in range(3):
                                d_c[ix](psit2_g, dpsit_g, phase_cd)
                                tmp[ix] = gd.integrate(psit1_g * dpsit_g)
                            rho_G[0] = -1j * np.dot(self.qq_v, tmp)

                        # PAW correction
                        for a, id in enumerate(calc.wfs.setups.id_a):
                            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                            gemv(1.0, self.phi_aGp[a], P_p, 1.0, rho_G)

                        if self.optical_limit:
                            rho_G[0] /= e_kn[ibzkpt2, m] - e_kn[ibzkpt1, n]

                        rho_GG = np.outer(rho_G, rho_G.conj())
                        
                        if not self.hilbert_trans:
                            for iw in range(self.Nw_local):
                                w = self.w_w[iw + self.wstart] / Hartree
                                C =  (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) / (
                                     w + e_kn[ibzkpt1, n] - e_kn[ibzkpt2, m] + 1j * self.eta)
                                axpy(C, rho_GG, chi0_wGG[iw])
                        else:
                            focc = f_kn[ibzkpt1,n] - f_kn[ibzkpt2,m]
                            w0 = e_kn[ibzkpt2,m] - e_kn[ibzkpt1,n]
                            scal(focc, rho_GG)

                            # calculate delta function
                            w0_id = int(w0 / self.dw)
                            if w0_id + 1 < self.NwS:
                                # rely on the self.NwS_local is equal in each node!
                                if self.wScomm.rank == w0_id // self.NwS_local:
                                    alpha = (w0_id + 1 - w0/self.dw) / self.dw
                                    axpy(alpha, rho_GG, specfunc_wGG[w0_id % self.NwS_local] )

                                if self.wScomm.rank == (w0_id+1) // self.NwS_local:
                                    alpha =  (w0 / self.dw - w0_id) / self.dw
                                    axpy(alpha, rho_GG, specfunc_wGG[(w0_id+1) % self.NwS_local] )

#                            deltaw = delta_function(w0, self.dw, self.NwS, self.sigma)
#                            for wi in range(self.NwS_local):
#                                if deltaw[wi + self.wS1] > 1e-8:
#                                    specfunc_wGG[wi] += tmp_GG * deltaw[wi + self.wS1]
                if self.nkpt == 1:
                    if n == 0:
                        dt = time() - t0
                        totaltime = dt * self.nband_local
                        self.printtxt('Finished n 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime) )
                    if rank == 0 and self.nband_local // 5 > 0:
                        if n > 0 and n % (self.nband_local // 5) == 0:
                            dt = time() - t0
                            self.printtxt('Finished n %d in %f seconds, estimated %f seconds left.'%(n, dt, totaltime-dt))
            if calc.wfs.world.size != 1:
                self.kcomm.barrier()            
            if k == 0:
                dt = time() - t0
                totaltime = dt * self.nkpt_local
                self.printtxt('Finished k 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime))
                
            if rank == 0 and self.nkpt_local // 5 > 0:            
                if k > 0 and k % (self.nkpt_local // 5) == 0:
                    dt =  time() - t0
                    self.printtxt('Finished k %d in %f seconds, estimated %f seconds left.  '%(k, dt, totaltime - dt) )
        self.printtxt('Finished summation over k')

        self.kcomm.barrier()
        del rho_GG, rho_G
        # Hilbert Transform
        if not self.hilbert_trans:
            self.kcomm.sum(chi0_wGG)
        else:
            self.kcomm.sum(specfunc_wGG)
            if self.wScomm.size == 1:
                chi0_wGG = hilbert_transform(specfunc_wGG, self.Nw, self.dw, self.eta)[self.wstart:self.wend]
                self.printtxt('Finished hilbert transform !')
                del specfunc_wGG
            else:
                # redistribute specfunc_wGG to all nodes
                assert self.NwS % size == 0
                NwStmp1 = (rank % self.kcomm.size) * self.NwS // size
                NwStmp2 = (rank % self.kcomm.size + 1) * self.NwS // size 
                specfuncnew_wGG = specfunc_wGG[NwStmp1:NwStmp2]
                del specfunc_wGG
                
                coords = np.zeros(self.wcomm.size, dtype=int)
                nG_local = self.npw**2 // self.wcomm.size
                if self.wcomm.rank == self.wcomm.size - 1:
                    nG_local = self.npw**2 - (self.wcomm.size - 1) * nG_local
                self.wcomm.all_gather(np.array([nG_local]), coords)
        
                specfunc_Wg = SliceAlongFrequency(specfuncnew_wGG, coords, self.wcomm)
                self.printtxt('Finished Slice Along Frequency !')
                chi0_Wg = hilbert_transform(specfunc_Wg, self.Nw, self.dw, self.eta)[:self.Nw]
                self.printtxt('Finished hilbert transform !')
                self.comm.barrier()
                del specfunc_Wg
        
                chi0_wGG = SliceAlongOrbitals(chi0_Wg, coords, self.wcomm)
                self.printtxt('Finished Slice along orbitals !')
                self.comm.barrier()
                del chi0_Wg
        
        self.chi0_wGG = chi0_wGG / self.vol

        self.printtxt('')
        self.printtxt('Finished chi0 !')

        return


    def get_wavefunction(self, ibzk, n, k, check_focc=True):

        if self.calc.wfs.kpt_comm.size != world.size or world.size == 1:

            if check_focc == False:
                return
            else:
                psit_G = self.calc.wfs.get_wave_function_array(n, ibzk, 0)
        
                if self.calc.wfs.world.size == 1:
                    return np.complex128(psit_G)
                
                if not self.calc.wfs.world.rank == 0:
                    psit_G = self.calc.wfs.gd.empty(dtype=self.calc.wfs.dtype,
                                                    global_array=True)
                self.calc.wfs.world.broadcast(psit_G, 0)
        
                return np.complex128(psit_G)
        else:
            if self.nkpt % size != 0:
                raise ValueError('The number of kpoints should be divided by the number of cpus for no wfs dumping mode ! ')

            # support ground state calculation with only kpoint parallelization
            kpt_rank, u = self.calc.wfs.kd.get_rank_and_index(0, ibzk)
            bzkpt_rank = k // self.nkpt_local
            
            klist = np.array([kpt_rank, u, bzkpt_rank])
            klist_kcomm = np.zeros((self.kcomm.size, 3), dtype=int)            
            self.kcomm.all_gather(klist, klist_kcomm)

            check_focc_global = np.zeros(self.kcomm.size, dtype=bool)
            self.kcomm.all_gather(np.array([check_focc]), check_focc_global)

            psit_G = self.calc.wfs.gd.empty(dtype=self.calc.wfs.dtype)
            
	    for i in range(self.kcomm.size):
                if check_focc_global[i] == True:
                    kpt_rank, u, bzkpt_rank = klist_kcomm[i]
                    if kpt_rank == bzkpt_rank:
                        if rank == kpt_rank:
                            psit_G = self.calc.wfs.kpt_u[u].psit_nG[n]
                    else:
                        if rank == kpt_rank:
                            world.send(self.calc.wfs.kpt_u[u].psit_nG[n],
                                       bzkpt_rank, 1300+bzkpt_rank)
                        if rank == bzkpt_rank:
                            psit_G = self.calc.wfs.gd.empty(dtype=self.calc.wfs.dtype)
                            world.receive(psit_G, kpt_rank, 1300+bzkpt_rank)
                    
            self.wScomm.broadcast(psit_G, 0)
            return psit_G
        
    def output_init(self):

        if self.txtname is None:
            if rank == 0:
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull

        else:
            assert type(self.txtname) is str
            from ase.parallel import paropen
            self.txt = paropen(self.txtname,'w')


    def parallel_init(self):
        """Parallel initialization. By default, only use kcomm and wcomm.

        Parameters:

            kcomm:
                 kpoint communicator
            wScomm:
                 spectral function communicator
            wcomm:
                 frequency communicator
        """

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
            rank = world.rank
            self.comm = world
        else:
            from gpaw.mpi import world, rank, size

        wcommsize = int(self.NwS * self.npw**2 * 16. / 1024**2) // 1500 # megabyte
        wcommsize += 1
        if size < wcommsize:
            raise ValueError('Number of cpus are not enough ! ')
        if self.kcommsize is None:
            self.kcommsize = world.size
        if wcommsize > size // self.kcommsize: # if matrix too large, overwrite kcommsize and distribute matrix
            self.printtxt('kcommsize is over written ! ')
            while size % wcommsize != 0:
                wcommsize += 1
            self.kcommsize = size // wcommsize
            assert self.kcommsize * wcommsize == size
            if self.kcommsize < 1:
                raise ValueError('Number of cpus are not enough ! ')

        self.kcomm, self.wScomm, self.wcomm = set_communicator(world, rank, size, self.kcommsize)

        if self.nkpt != 1:
            self.nkpt, self.nkpt_local, self.kstart, self.kend = parallel_partition(
                               self.nkpt, self.kcomm.rank, self.kcomm.size, reshape=False)
            self.nband_local = self.nbands
            self.nstart = 0
            if self.hilbert_trans:
                self.nend = self.nvalbands
            else:
                self.nend = self.nbands
        else:
            # if number of kpoints == 1, use band parallelization
            self.nkpt_local = 1
            self.kstart = 0
            self.kend = 1

            self.nvalbands, self.nband_local, self.nstart, self.nend = parallel_partition(
                               self.nvalbands, self.kcomm.rank, self.kcomm.size, reshape=False)

        if self.NwS % size != 0:
            self.NwS -= self.NwS % size
            
        self.NwS, self.NwS_local, self.wS1, self.wS2 = parallel_partition(
                               self.NwS, self.wScomm.rank, self.wScomm.size, reshape=False)

        if self.hilbert_trans:
            self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=True)
        else:
            if self.Nw > 1:
                assert self.Nw % (size / self.kcomm.size) == 0
                self.wcomm = self.wScomm
                self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=False)
            else:
                # if frequency point is too few, then dont parallelize
                self.wcomm = serial_comm
                self.wstart = 0
                self.wend = self.Nw
                self.Nw_local = self.Nw

        return


    def printtxt(self, text):
        print >> self.txt, text


    def print_stuff(self):

        printtxt = self.printtxt
        printtxt('')
        printtxt('Parameters used:')
        printtxt('')
        printtxt('Unit cell (a.u.):')
        printtxt(self.acell_cv)
        printtxt('Reciprocal cell (1/a.u.)')
        printtxt(self.bcell_cv)
        printtxt('Grid spacing (a.u.):')
        printtxt(self.h_cv)
        printtxt('Number of Grid points / G-vectors, and in total: (%d %d %d), %d'
                  %(self.nG[0], self.nG[1], self.nG[2], self.nG0))
        printtxt('Volome of cell (a.u.**3)     : %f' %(self.vol) )
        printtxt('BZ volume (1/a.u.**3)        : %f' %(self.BZvol) )
        printtxt('')                         
        printtxt('Number of bands              : %d' %(self.nbands) )
        printtxt('Number of kpoints            : %d' %(self.nkpt) )
        printtxt('Planewave ecut (eV)          : (%f, %f, %f)' %(self.ecut[0]*Hartree,self.ecut[1]*Hartree,self.ecut[2]*Hartree) )
        printtxt('Number of planewave used     : %d' %(self.npw) )
        printtxt('Broadening (eta)             : %f' %(self.eta * Hartree))
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        if self.hilbert_trans:
            printtxt('Number of specfunc points    : %d' % (self.NwS))
        printtxt('')
        if self.optical_limit:
            printtxt('Optical limit calculation ! (q=0.00001)')
        else:
            printtxt('q in reduced coordinate        : (%f %f %f)' %(self.q_c[0], self.q_c[1], self.q_c[2]) )
            printtxt('q in cartesian coordinate (1/A): (%f %f %f) '
                  %(self.qq_v[0] / Bohr, self.qq_v[1] / Bohr, self.qq_v[2] / Bohr) )
            printtxt('|q| (1/A)                      : %f' %(sqrt(np.dot(self.qq_v / Bohr, self.qq_v / Bohr))) )
        printtxt('')
        printtxt('Use Hilbert Transform: %s' %(self.hilbert_trans) )
        printtxt('')
        printtxt('Parallelization scheme:')
        printtxt('     Total cpus      : %d' %(self.comm.size))
        if self.nkpt == 1:
            printtxt('     nbands parsize  : %d' %(self.kcomm.size))
        else:
            printtxt('     kpoint parsize  : %d' %(self.kcomm.size))
        if self.hilbert_trans:
            printtxt('     specfunc parsize: %d' %(self.wScomm.size))
        printtxt('     w parsize       : %d' %(self.wcomm.size))
        printtxt('')
        printtxt('Memory usage estimation:')
        printtxt('     chi0_wGG        : %f M / cpu' %(self.Nw_local * self.npw**2 * 16. / 1024**2) )
        if self.hilbert_trans:
            printtxt('     specfunc_wGG    : %f M / cpu' %(self.NwS_local *self.npw**2 * 16. / 1024**2) )



