import sys
from time import time, ctime
import numpy as np
import gpaw.fftw as fftw
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw import extra_parameters
from gpaw.utilities.blas import gemv, scal, axpy, czher, ccher
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.fd_operators import Gradient
from gpaw.response.math_func import hilbert_transform
from gpaw.response.parallel import set_communicator, \
     parallel_partition, parallel_partition_list, SliceAlongFrequency, SliceAlongOrbitals
from gpaw.response.kernel import calculate_Kxc, calculate_Kc
from gpaw.utilities.memory import maxrss
from gpaw.response.base import BaseChi
import _gpaw
from gpaw.response.timing import Timer
from gpaw.response.cuda import BaseCuda

class Chi(BaseChi):
    """This class is a calculator for the linear density response function.

    Parameters:

        nband: int
            Number of bands.
        wmax: floadent
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
                 eshift=None,
                 ecut=10.,
                 smooth_cut=None,
                 density_cut=None,
                 G_plus_q=False,
                 eta=0.2,
                 rpad=np.array([1,1,1]),
                 vcut=None,
                 ftol=1e-5,
                 txt=None,
                 xc='ALDA',
                 hilbert_trans=True,
                 full_response=False,
                 optical_limit=False,
                 cell=None,
                 cuda=False,
                 nmultix=1,
                 sync=False,
                 comm=None,
                 kcommsize=None):

        BaseChi.__init__(self, calc=calc, nbands=nbands, w=w, q=q,
                         eshift=eshift, ecut=ecut, smooth_cut=smooth_cut,
                         density_cut=density_cut, G_plus_q=G_plus_q, eta=eta,
                         rpad=rpad, ftol=ftol, txt=txt,
                         optical_limit=optical_limit, cell=cell)
        
        self.xc = xc
        self.hilbert_trans = hilbert_trans
        self.full_hilbert_trans = full_response
        self.vcut = vcut
        self.cuda = cuda
        self.nmultix = nmultix
        self.sync = sync
        self.kcommsize = kcommsize
        self.comm = comm
        if self.comm is None:
            self.comm = world
        self.chi0_wGG = None

        
    def initialize(self, simple_version=False):

        self.printtxt('')
        self.printtxt('-----------------------------------------')
        self.printtxt('Response function calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        BaseChi.initialize(self)

        # Frequency init
        self.dw = None
        if len(self.w_w) == 1:
            self.hilbert_trans = False

        if self.hilbert_trans:
            self.dw = self.w_w[1] - self.w_w[0]
#            assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
            assert self.w_w.max() == self.w_w[-1]
            
            self.dw /= Hartree
            self.w_w  /= Hartree
            self.wmax = self.w_w[-1] 
            self.wcut = self.wmax + 5. / Hartree
#            self.Nw  = int(self.wmax / self.dw) + 1
            self.Nw = len(self.w_w)
            self.NwS = int(self.wcut / self.dw) + 1
        else:
            self.Nw = len(self.w_w)
            self.NwS = 0
            if len(self.w_w) > 2:
                self.dw = self.w_w[1] - self.w_w[0]
                assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all()
                self.dw /= Hartree

        self.nvalbands = self.nbands
        tmpn = np.zeros(self.nspins, dtype=int)
        for spin in range(self.nspins):
            for n in range(self.nbands):
                if (self.f_skn[spin][:, n] - self.ftol < 0).all():
                    tmpn[spin] = n
                    break
        if tmpn.max() > 0:
            self.nvalbands = tmpn.max()

        # Parallelization initialize
        self.parallel_init()

        # Printing calculation information
        self.print_chi()

        if extra_parameters.get('df_dry_run'):
            raise SystemExit

        calc = self.calc

        # For LCAO wfs
        if calc.input_parameters['mode'] == 'lcao':
            calc.initialize_positions()        
        self.printtxt('     Max mem sofar   : %f M / cpu' %(maxrss() / 1024**2))

        if simple_version is True:
            return
        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        self.phi_aGp = self.get_phi_aGp()
        self.printtxt('Finished phi_aGp !')
        mem = np.array([self.phi_aGp[i].size * 16 /1024.**2 for i in range(len(self.phi_aGp))])
        self.printtxt('     Phi_aGp         : %f M / cpu' %(mem.sum()))

        # Calculate ALDA kernel (not used in chi0)
        R_av = calc.atoms.positions / Bohr
        if self.xc == 'RPA': #type(self.w_w[0]) is float:
            self.Kc_GG = None
            self.printtxt('RPA calculation.')
        elif self.xc == 'ALDA' or self.xc == 'ALDA_X':
            Kc_G = calculate_Kc(self.q_c, self.Gvec_Gc, self.acell_cv,
                                  self.bcell_cv, self.calc.atoms.pbc, self.optical_limit, self.vcut)
            self.Kc_GG = np.outer(Kc_G, Kc_G)

            nt_sg = calc.density.nt_sG
            if (self.rpad > 1).any() or (self.pbc - True).any():
                nt_sG = np.zeros([self.nspins, self.nG[0], self.nG[1], self.nG[2]])
                for s in range(self.nspins):
                    nt_G = self.pad(nt_sg[s])
                    nt_sG[s] = nt_G
            else:
                nt_sG = nt_sg
            
            self.Kxc_sGG = calculate_Kxc(self.gd, # global grid
                                         nt_sG,
                                         self.npw, self.Gvec_Gc,
                                         self.nG, self.vol,
                                         self.bcell_cv, R_av,
                                         calc.wfs.setups,
                                         calc.density.D_asp,
                                         functional=self.xc,
                                         density_cut=self.density_cut)
            
            self.printtxt('Finished %s kernel ! ' % self.xc)

        return


    def calculate(self, seperate_spin=None):
        """Calculate the non-interacting density response function. """

        calc = self.calc
        kd = self.kd
        gd = self.gd
        sdisp_cd = gd.sdisp_cd
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        pt = self.pt
        f_skn = self.f_skn
        e_skn = self.e_skn
        nmultix = self.nmultix
        C_uw = np.zeros((nmultix, self.npw))
        spos_ac = calc.atoms.get_scaled_positions()

        # Matrix init
        sizeofdata = 16
        chi0_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype=complex)
        sizeofint = 4

        if self.cuda:
            print >> self.txt, 'Use Cuda ! '
            cu = BaseCuda()
            cu.chi_init(self, chi0_wGG)
            cu.paw_init(calc.wfs, spos_ac, self)

        if self.hilbert_trans:
            specfunc_wGG = np.zeros((self.NwS_local, self.npw, self.npw), dtype = complex)

        # Prepare for the derivative of pseudo-wavefunction
        if self.optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

        # fftw init
        fft_R = fftw.empty(self.nG, complex)
        fft_G = fft_R
        fftplan = fftw.FFTPlan(fft_R, fft_G, -1, fftw.ESTIMATE)

        if fftw.FFTPlan is fftw.NumpyFFTPlan:
            self.printtxt('Using Numpy FFT.')
        else:
            self.printtxt('Using FFTW Library.') 

        use_zher = False
        if self.eta < 1e-3:
            use_zher = True

        rho_G = np.zeros(self.npw, dtype=complex)
        t0 = time()

        if seperate_spin is None:
            spinlist = np.arange(self.nspins)
        else:
            spinlist = [seperate_spin]
        
        if self.sync: # perform timing when synchronize
#            assert self.cuda is True
            print >> self.txt, 'Initialization time', time() - self.starttime
            timer = Timer()
            self.timer = timer
        
        for spin in spinlist:
            if not (f_skn[spin] > self.ftol).any():
                self.chi0_wGG = chi0_wGG
                continue
            
            for k in range(self.kstart, self.kend):
#                if k > 0:
#                    XX
                k_pad = False
                if k >= self.nkpt:
                    k = 0
                    k_pad = True
    
                # Find corresponding kpoint in IBZ
                ibzkpt1 = kd.bz2ibz_k[k]
                if self.optical_limit:
                    ibzkpt2 = ibzkpt1
                else:
                    ibzkpt2 = kd.bz2ibz_k[kq_k[k]]

                if self.sync: timer.start('wfs_transform')
                if self.pwmode:
                    N_c = self.gd.N_c
                    k_c = self.kd.ibzk_kc[ibzkpt1]
                    eikr1_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k_c / N_c).T)
                    k_c = self.kd.ibzk_kc[ibzkpt2]
                    eikr2_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k_c / N_c).T)
                    
                index1_g, phase1_g = kd.get_transform_wavefunction_index(self.nG, k)
                index2_g, phase2_g = kd.get_transform_wavefunction_index(self.nG, kq_k[k])

                if self.sync: timer.end('wfs_transform')

                if self.pwmode and self.cuda:
                    cu.kspecific_init(self, spin, k, ibzkpt1, ibzkpt2, index1_g, index2_g)

                for n in range(self.nvalbands):
                    if self.calc.wfs.world.size == 1:
                        if (self.f_skn[spin][ibzkpt1, n] - self.ftol < 0):
                            continue

                    t1 = time()
                    if self.sync: timer.start('wfs_read')
                    if not self.pwmode:
                        psitold_g = self.get_wavefunction(ibzkpt1, n, True, spin=spin)
                    else:
                        if not self.cuda:
                            u = self.kd.get_rank_and_index(spin, ibzkpt1)[1]
                            psitold_g = calc.wfs._get_wave_function_array(u, n, realspace=True,
                                                                          phase=eikr1_R)
                        else:
                            if self.sync: timer.start('wfs_get')
                            cu.get_wfs(calc.wfs.kpt_u[cu.u1].psit_nG[n], cu.psit_G, cu.Q1_G, cu.ncoef, 1, self)
                            if self.sync: timer.end('wfs_get')
                            _gpaw.cuCopy_vector(cu.tmp_uQ, cu.tmp_Q, self.nG0) 

                    if self.sync: timer.end('wfs_read')

                    if self.sync: timer.start('wfs_transform')
                    if not self.cuda:
                        psit1new_g_tmp = kd.transform_wave_function(psitold_g,k,index1_g,phase1_g)
                        if (self.rpad > 1).any() or (self.pbc - True).any():
                            psit1new_g = self.pad(psit1new_g_tmp)
                        else:
                            psit1new_g = psit1new_g_tmp
                    else:
                        # dev_psit1_R is the wave function of (k, n) on device
                        cu.trans_wfs(cu.tmp_Q, cu.psit1_R, cu.ind.index1_Q, cu.trans1, cu.time_rev1, 1)
                        # padding does not work on cuda now !! 

                    if self.sync: timer.end('wfs_transform')

                   # PAW part
                    if self.sync: timer.start('paw')
                    if not self.cuda:
                        if not self.pwmode:
                            if (calc.wfs.world.size > 1 or self.nkpt==1):
                                P1_ai = pt.dict()
                                pt.integrate(psit1new_g, P1_ai, k)
                            else:
                                P1_ai = self.get_P_ai(k, n, spin)
                        else:
                            if calc.wfs.kpt_u[0].P_ani is None: 
                                # first calculate P_ai at ibzkpt, then rotate to k
                                Ptmp_ai = pt.dict()
                                kpt = calc.wfs.kpt_u[u]
                                pt.integrate(kpt.psit_nG[n], Ptmp_ai, ibzkpt1)
                                P1_ai = self.get_P_ai(k, n, spin, Ptmp_ai)
                            else:
                                P1_ai = self.get_P_ai(k, n, spin)
                    else:
                        _gpaw.cuMemset(cu.mlocallist, 0, sizeofint*nmultix)
                        _gpaw.cuSetVector(1,sizeofint,np.array(n, np.int32),1,cu.mlocallist,1)      
                        cu.get_P_ai(cu.P_P1_ani, cu.P_P1_ai, cu.P1_ani, cu.P1_ai, 
                                            cu.time_rev1, cu.s1, ibzkpt1, cu.mlocallist, 1)

                    if self.sync: timer.end('paw')

                    if not self.cuda: # or (self.cuda and self.optical_limit):
                        psit1_g = psit1new_g.conj() * self.expqr_g # cublas optical_limit is wrong now

                    imultix = 0
                    for m in self.mlist:
                        if self.nbands > 50 and m % 50 == 0:
                            if not self.sync:
                                print >> self.txt, k, n, m, time() - t0
                            else:
                                print >> self.txt, '\n', k, n, m, time() - t0, time() - t0 - timer.get_tot_timing()
                                for key in timer.timers.keys():
                                    print >> self.txt, '%15s'%(key), timer.get_timing(key)
    		    
                        check_focc = (f_skn[spin][ibzkpt1, n] - f_skn[spin][ibzkpt2, m]) > self.ftol
    
                        if not self.pwmode:
                            psitold_g = self.get_wavefunction(ibzkpt2, m, check_focc, spin=spin)
    
                        if check_focc:                            

                            if self.sync: timer.start('wfs_read')
                            if self.pwmode:
                                if not self.cuda:
                                    u = self.kd.get_rank_and_index(spin, ibzkpt2)[1]
                                    psitold_g = calc.wfs._get_wave_function_array(u, m, realspace=True, phase=eikr2_R)
                                else:
                                    # dev_psit2_R is the (transformed) wave function for (kq[k],m) on device
                                    if imultix == 0:
                                        mlocallist = []
                                        iu = 0
                                        if self.sync: timer.start('wfs_read_disk')
                                        psit_uG = np.zeros((nmultix, cu.ncoef2),complex)
                                        while iu < nmultix  and m + iu < self.nbands:
                                            if (f_skn[spin][ibzkpt1, n] - f_skn[spin][ibzkpt2, m+iu]) > self.ftol:
                                                psit_uG[iu] = calc.wfs.kpt_u[cu.u2].psit_nG[m+iu]
                                                mlocallist.append(m+iu)
                                                iu += 1
                                        if self.sync: timer.end('wfs_read_disk')
                                        if self.sync: timer.start('wfs_get')
                                        cu.get_wfs(psit_uG, cu.psit_uG, cu.Q2_G, cu.ncoef2, nmultix, self)
                                        if self.sync: timer.end('wfs_get')
                                        _gpaw.cuMemset(cu.mlocallist, 0, sizeofint*nmultix)
                                        _gpaw.cuSetVector(len(mlocallist),sizeofint,np.array(mlocallist, np.int32),1,
                                                          cu.mlocallist,1)                                    

                            if self.sync: timer.end('wfs_read')

                            if self.sync: timer.start('wfs_transform')
                            if not self.cuda:
                                psit2_g_tmp = kd.transform_wave_function(psitold_g, kq_k[k], index2_g, phase2_g)
                                if (self.rpad > 1).any() or (self.pbc - True).any():
                                    psit2_g = self.pad(psit2_g_tmp)
                                else:
                                    psit2_g = psit2_g_tmp
                            else:
                                if imultix == 0:
                                    # dev_psit2_R is the wave function of (k+q, m) on device
                                    cu.trans_wfs(cu.tmp_uQ, cu.psit2_uR, cu.ind.index2_Q, cu.trans2, cu.time_rev2, nmultix)

                            if self.sync: timer.end('wfs_transform')

                            if self.cuda:
                                GPUpointer = imultix * self.npw * sizeofdata
                            # fft
                            if self.sync: timer.start('fft')
                            if not self.cuda:
                                fft_R[:] = psit2_g * psit1_g

                                fftplan.execute()
                                fft_G *= self.vol / self.nG0
        #                        tmp_g = np.fft.fftn(psit2_g*psit1_g) * self.vol / self.nG0
                            else:
                                # dev_psit1_R is re-used ! can't change !

                                # calculate cu.psit1_R.conj() * cu.expqr_R * cu.psit2_uR 
                                
                                if self.optical_limit: # psit2_R has to be saved for later
                                    if imultix == 0:
                                        _gpaw.cuCopy_vector(cu.psit2_uR, cu.optpsit2_uR, self.nG0*nmultix) 
                                    _gpaw.cuCopy_vector(cu.optpsit2_uR+imultix*sizeofdata*cu.nG0, cu.optpsit2_R, self.nG0)
                                    
                                if imultix == 0:
                                    _gpaw.cuDensity_matrix_R(cu.psit1_R, cu.expqr_R, cu.psit2_uR, self.nG0, nmultix)
                                    _gpaw.cufft_execZ2Z(cu.cufftplanmany, cu.psit2_uR, 
                                                    cu.psit2_uR, -1) # R->Q
                                    _gpaw.cuZscal(cu.handle, self.nG0*nmultix, self.vol/self.nG0, cu.psit2_uR, 1)

                            if self.sync: timer.end('fft')



                            if self.sync: timer.start('mapG')
                            if not self.cuda:
                                rho_G = fft_G.ravel()[self.Gindex_G]
                            else:
                                if imultix == 0:
                                    _gpaw.cuMap_Q2G(cu.psit2_uR, cu.rho_uG, 
                                                cu.ind.Gindex_G, self.npw, self.nG0, nmultix)

                            if self.sync: timer.end('mapG')

                            if self.sync: timer.start('opt')
                            if self.optical_limit:
                                if not self.cuda:
                                    phase_cd = np.exp(2j * pi * sdisp_cd * bzk_kc[kq_k[k], :, np.newaxis])
                                    for ix in range(3):
                                        d_c[ix](psit2_g, dpsit_g, phase_cd)
                                        if self.cuda:
                                            tmp[ix] = gd.integrate(psit1_g * dpsit_g * eikr_R.conj())
                                        else:
                                            tmp[ix] = gd.integrate(psit1_g * dpsit_g)
                                    rho_G[0] = -1j * np.dot(self.qq_v, tmp)
                                else:
                                    rhoG0 = cu.calculate_opt(cu.psit1_R)
                                    status = _gpaw.cuSetVector(1, sizeofdata, rhoG0, 1, cu.rho_uG+GPUpointer, 1)
                            if self.sync: timer.end('opt')

                            if self.sync: timer.start('paw')
                            # PAW correction
                            if not self.cuda:
                                if not self.pwmode:
                                    if (calc.wfs.world.size > 1 or self.nkpt==1):
                                        P2_ai = pt.dict()
                                        pt.integrate(psit2_g, P2_ai, kq_k[k])
                                    else:
                                        P2_ai = self.get_P_ai(kq_k[k], m, spin)                    
                                else:
                                    if calc.wfs.kpt_u[0].P_ani is None: 
                                        Ptmp_ai = pt.dict()
                                        kpt = calc.wfs.kpt_u[u]
                                        pt.integrate(kpt.psit_nG[m], Ptmp_ai, ibzkpt2)
                                        P2_ai = self.get_P_ai(kq_k[k], m, spin, Ptmp_ai)
                                    else:
                                        P2_ai = self.get_P_ai(kq_k[k], m, spin)         
                            else:
                                if imultix == 0:
                                    cu.get_P_ai(cu.P_P2_ani, cu.P_P2_aui, cu.P2_ani, cu.P2_aui, 
                                                cu.time_rev2, cu.s2, ibzkpt2, cu.mlocallist, len(mlocallist))

                            if self.sync: timer.end('paw')

                            for a, id in enumerate(calc.wfs.setups.id_a):
                                if self.sync: timer.start('paw_outer')
                                if not self.cuda:
                                    P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                                else:
                                    if imultix == 0:
                                        cu.get_P_aup(cu.P1_ai, cu.P2_aui, cu.P_aup, len(mlocallist))
                                if self.sync: timer.end('paw_outer')

                                if self.sync: timer.start('cugemv')
                                if not self.cuda:
                                    gemv(1.0, self.phi_aGp[a], P_p, 1.0, rho_G)
                                else:
                                    if imultix == 0:
                                        Ni = cu.host_Ni_a[a]
                                        _gpaw.cugemm(cu.handle,1.0,  cu.P_phi_aGp[a], cu.P_P_aup[a], 1.0, cu.rho_uG, \
                                                     nmultix, self.npw, Ni*Ni, Ni*Ni, Ni*Ni, self.npw, 0, 1)
                                if self.sync: timer.end('cugemv')


                            if self.optical_limit and self.cuda:
                                if self.sync: timer.start('cugemv')
                                status = _gpaw.cuGetVector(1,sizeofdata,cu.rho_uG+GPUpointer,1,rho_G,1)
                                if self.sync: timer.end('cugemv')

                            if self.optical_limit:
                                if self.sync: timer.start('opt')
                                if np.abs(self.enoshift_skn[spin][ibzkpt2, m] -
                                          self.enoshift_skn[spin][ibzkpt1, n]) > 0.1/Hartree:
                                    rho_G[0] /= self.enoshift_skn[spin][ibzkpt2, m] \
                                                - self.enoshift_skn[spin][ibzkpt1, n]
                                else:
                                    rho_G[0] = 0.
                                if self.sync: timer.end('opt')
    
                            if k_pad:
                                rho_G[:] = 0.
    
                            if not self.hilbert_trans:
                                if not use_zher:
                                    rho_GG = np.outer(rho_G, rho_G.conj())
                                else:
                                    if self.sync: timer.start('cugemv')
                                    if self.optical_limit and self.cuda:
                                        status = _gpaw.cuSetVector(1, sizeofdata, rho_G, 1, cu.rho_uG+GPUpointer, 1)
                                    if self.sync: timer.end('cugemv')

                                for iw in range(self.Nw_local):
                                    w = self.w_w[iw + self.wstart] / Hartree
                                    coef = ( 1. / (w + e_skn[spin][ibzkpt1, n] - e_skn[spin][ibzkpt2, m]
                                                   + 1j * self.eta) 
                                           - 1. / (w - e_skn[spin][ibzkpt1, n] + e_skn[spin][ibzkpt2, m]
                                                   + 1j * self.eta) )
                                    C =  (f_skn[spin][ibzkpt1, n] - f_skn[spin][ibzkpt2, m]) * coef
                                    if self.cuda:
                                        C_uw[imultix, iw] = C.real
                                        assert C.real <=0

                                    if self.sync: timer.start('zherk')
                                    if use_zher:
                                        if not self.cuda:
                                            czher(C.real, rho_G.conj(), chi0_wGG[iw])
                                        else:
                                            if imultix == nmultix - 1 or m == len(self.mlist) - 1:
                                                matrixGPU_GG = cu.chi0_w[iw]
                                                # replace with zherk
#                                                status = _gpaw.cuZher(handle,0,self.npw,C.real,GPU_rho_G,1,
#                                                                  matrixGPU_GG,self.npw)
                                                if iw == 0:
                                                    alpha_u = np.sqrt(-C_uw[:,iw])
                                                else:
                                                    alpha_u = np.sqrt(C_uw[:,iw]/C_uw[:,iw-1])
                                                
                                                for iu in range(nmultix):
                                                    alpha = alpha_u[iu]
                                                    _gpaw.cuZscal(cu.handle,self.npw,alpha,cu.rho_uG+iu*self.npw*sizeofdata,1)

                                                status = _gpaw.cuZherk(cu.handle,0,self.npw,nmultix,-1.0,cu.rho_uG,self.npw,1.0,
                                                                       matrixGPU_GG,self.npw)
                                    else:
                                        axpy(C, rho_GG, chi0_wGG[iw])
                                    if self.sync: timer.end('zherk')

                                if self.sync: timer.start('zherk')
                                if self.cuda:
                                    if imultix == nmultix - 1:
                                        imultix = 0
                                        _gpaw.cuMemset(cu.rho_uG, 0, sizeofdata*nmultix*self.npw)
                                    else:
                                        imultix += 1
                                if self.sync: timer.end('zherk')

                            else:
                                rho_GG = np.outer(rho_G, rho_G.conj())
                                focc = f_skn[spin][ibzkpt1,n] - f_skn[spin][ibzkpt2,m]
                                w0 = e_skn[spin][ibzkpt2,m] - e_skn[spin][ibzkpt1,n]
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
                            totaltime = dt * self.nvalbands * self.nspins
                            self.printtxt('Finished n 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime) )
                        if rank == 0 and self.nvalbands // 5 > 0:
                            if n > 0 and n % (self.nvalbands // 5) == 0:
                                dt = time() - t0
                                self.printtxt('Finished n %d in %f seconds, estimated %f seconds left.'%(n, dt, totaltime-dt))

                if self.cuda:
                    cu.kspecific_free(self)
                if calc.wfs.world.size != 1:
                    self.kcomm.barrier()            
                if k == 0:
                    dt = time() - t0
                    totaltime = dt * self.nkpt_local * self.nspins
                    self.printtxt('Finished k 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime))
                    
                if rank == 0 and self.nkpt_local // 5 > 0:            
                    if k > 0 and k % (self.nkpt_local // 5) == 0:
                        dt =  time() - t0
                        self.printtxt('Finished k %d in %f seconds, estimated %f seconds left.  '%(k, dt, totaltime - dt) )
        self.printtxt('Finished summation over k')

        self.kcomm.barrier()
        
        # Hilbert Transform
        if not self.hilbert_trans:
            if not use_zher: # in fact, it should be, if not use_cuzher
                for iw in range(self.Nw_local):
                    self.kcomm.sum(chi0_wGG[iw])
            else:
                if self.cuda:
                    for iw in range(self.Nw_local):
                        status = _gpaw.cuGetMatrix(self.npw,self.npw,sizeofdata,cu.chi0_w[iw],self.npw,
                                                   chi0_wGG[iw],self.npw)
                        chi0_wGG[iw] = chi0_wGG[iw].conj()

                    cu.chi_free(self)
                    cu.paw_free()
                    
                for iw in range(self.Nw_local):
                    self.kcomm.sum(chi0_wGG[iw])

                assert (np.abs(chi0_wGG[0,1:,0]) < 1e-10).all()
                for iw in range(self.Nw_local):
                    chi0_wGG[iw] += chi0_wGG[iw].conj().T
                    for iG in range(self.npw):
                        chi0_wGG[iw, iG, iG] /= 2.
#                        assert np.abs(np.imag(chi0_wGG[iw, iG, iG])) < 1e-6

        else:
            for iw in range(self.NwS_local):
                self.kcomm.sum(specfunc_wGG[iw])
            if self.wScomm.size == 1:
                chi0_wGG = hilbert_transform(specfunc_wGG, self.w_w, self.Nw, self.dw, self.eta,
                                             self.full_hilbert_trans)[self.wstart:self.wend]
                self.printtxt('Finished hilbert transform !')
                del specfunc_wGG
            else:
                # redistribute specfunc_wGG to all nodes
                size = self.comm.size
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
                chi0_Wg = hilbert_transform(specfunc_Wg, self.w_w, self.Nw, self.dw, self.eta,
                                            self.full_hilbert_trans)[:self.Nw]
                self.printtxt('Finished hilbert transform !')
                self.comm.barrier()
                del specfunc_Wg
        
                chi0_wGG = SliceAlongOrbitals(chi0_Wg, coords, self.wcomm)
                self.printtxt('Finished Slice along orbitals !')
                self.comm.barrier()
                del chi0_Wg
        
        self.chi0_wGG = chi0_wGG
        self.chi0_wGG /= self.vol

        if self.smooth_cut is not None:
            for iw in range(self.Nw_local):
                self.chi0_wGG[iw] *= np.outer(self.G_weights, self.G_weights)
            
        self.printtxt('')
        self.printtxt('Finished chi0 !')

        return


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
            world = self.comm
            rank = self.comm.rank
            size = self.comm.size

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

        if self.nkpt % world.size ==0:
            self.nkpt_reshape = self.nkpt
            self.nkpt_reshape, self.nkpt_local, self.kstart, self.kend = parallel_partition(
                               self.nkpt_reshape, self.kcomm.rank, self.kcomm.size, reshape=True, positive=True)
            self.mband_local = self.nvalbands
            self.mlist = np.arange(self.nbands)
        else:
            # if number of kpoints == 1, use band parallelization
            self.nkpt_local = self.nkpt
            self.kstart = 0
            self.kend = self.nkpt
            self.nkpt_reshape = self.nkpt

            self.nbands, self.mband_local, self.mlist = parallel_partition_list(
                               self.nbands, self.kcomm.rank, self.kcomm.size)

        if self.NwS % size != 0:
            self.NwS -= self.NwS % size
            
        self.NwS, self.NwS_local, self.wS1, self.wS2 = parallel_partition(
                               self.NwS, self.wScomm.rank, self.wScomm.size, reshape=False)

        if self.hilbert_trans:
            self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=True)
        else:
            if self.Nw > 1:
#                assert self.Nw % (self.comm.size / self.kcomm.size) == 0
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

    def print_chi(self):

        printtxt = self.printtxt
        printtxt('Use Hilbert Transform: %s' %(self.hilbert_trans) )
        printtxt('Calculate full Response Function: %s' %(self.full_hilbert_trans) )
        printtxt('')
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        if self.hilbert_trans:
            printtxt('Number of specfunc points    : %d' % (self.NwS))
        printtxt('')
        printtxt('Parallelization scheme:')
        printtxt('     Total cpus      : %d' %(self.comm.size))
        if self.nkpt == 1:
            printtxt('     nbands parsize  : %d' %(self.kcomm.size))
        else:
            printtxt('     kpoint parsize  : %d' %(self.kcomm.size))
            if self.nkpt_reshape > self.nkpt:
                self.printtxt('        kpoints (%d-%d) are padded with zeros' %(self.nkpt,self.nkpt_reshape))

        if self.hilbert_trans:
            printtxt('     specfunc parsize: %d' %(self.wScomm.size))
        printtxt('     w parsize       : %d' %(self.wcomm.size))
        printtxt('')
        printtxt('Memory usage estimation:')
        printtxt('     chi0_wGG        : %f M / cpu' %(self.Nw_local * self.npw**2 * 16. / 1024**2) )
        if self.hilbert_trans:
            printtxt('     specfunc_wGG    : %f M / cpu' %(self.NwS_local *self.npw**2 * 16. / 1024**2) )

