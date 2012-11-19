import numpy as np
import pickle
from math import pi, sqrt
from time import time, ctime
from datetime import timedelta
from ase.parallel import paropen
from ase.units import Hartree, Bohr
from gpaw import GPAW
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.utilities.blas import gemmdot
from gpaw.utilities.memory import maxrss
from gpaw.xc.tools import vxc
from gpaw.wavefunctions.pw import PWWaveFunctions
from gpaw.response.parallel import set_communicator, parallel_partition, SliceAlongFrequency, GatherOrbitals
from gpaw.response.base import BASECHI

class GW(BASECHI):

    def __init__(
                 self,
                 file=None,
                 nbands=None,
                 bands=None,
                 kpoints=None,
                 w=None,
                 ecut=150.,
                 eta=0.1,
                 ppa=False,
                 E0=None,
                 hilbert_trans=False,
                 wpar=1,
                 vcut=None,
                 exxfile=None,
                 txt=None
                ):

        static=False

        if ppa: # Plasmon Pole Approximation
            w_w = (0.,)
            hilbert_trans=False
            wpar=1
            if E0 is None:
                E0 = Hartree
        elif w==None: # static COHSEX
            w_w = (0.,)
            static=True
            hilbert_trans=False
            wpar=1
        else:
            # create nonlinear frequency grid
            # grid is linear from 0 to wcut with spacing dw
            # spacing is linearily increasing between wcut and wmax
            # Hilbert transforms are still carried out on linear grid
            wcut = w[0]
            wmax = w[1]
            dw = w[2]
            w_w = np.linspace(0., wcut, wcut/dw+1)
            i=1
            wi=wcut
            while wi < wmax:
                wi += i*dw
                w_w = np.append(w_w, wi)
                i+=1
            while len(w_w) % wpar != 0:
                wi += i*dw
                w_w = np.append(w_w, wi)
                i+=1

            dw_w = np.zeros(len(w_w))
            dw_w[0] = dw
            dw_w[1:] = w_w[1:] - w_w[:-1]

            self.dw_w = dw_w
            self.eta_w = dw_w * 4
            self.wcut = wcut

        BASECHI.__init__(self, calc=file, nbands=nbands, w=w, ecut=ecut, eta=eta, txt=txt)

        self.file = file
        self.vcut = vcut
        self.bands = bands
        self.kpoints = kpoints
        self.hilbert_trans = hilbert_trans
        self.wpar = wpar
        self.exxfile = exxfile
        self.w_w = w_w
        self.ppa = ppa
        self.E0 = E0
        self.static = static

    def initialize(self):

        self.printtxt('-----------------------------------------------')
        self.printtxt('GW calculation started at: \n')
        self.printtxt(ctime())
        self.starttime = time()
        
        BASECHI.initialize(self)
        calc = self.calc
        self.kd = kd = self.calc.wfs.kd
        self.nkpt = kd.nbzkpts
        self.nikpt = kd.nibzkpts

        # q point init
        self.bzq_kc = kd.get_bz_q_points()
        self.ibzq_qc = self.bzq_kc # q point symmetry is not used at the moment.
        self.nqpt = np.shape(self.bzq_kc)[0]
        
        # frequency points init
        if not self.ppa and not self.static:
            self.dw_w /= Hartree
            self.w_w  /= Hartree
            self.eta_w /= Hartree
            self.wmax = self.w_w[-1]
            self.wmin = self.w_w[0]
            self.dw = self.w_w[1] - self.w_w[0]
            self.Nw = len(self.w_w)

        # eigenvalues and occupations init
        self.e_skn = np.zeros((self.nspins, self.nikpt, calc.wfs.bd.nbands), dtype=float)
        self.f_skn = np.zeros((self.nspins, self.nikpt, calc.wfs.bd.nbands), dtype=float)
        for s in range(self.nspins):
            for k in range(self.nikpt):
                self.e_skn[s,k] = calc.get_eigenvalues(kpt=k, spin=s) / Hartree
                self.f_skn[s,k] = calc.get_occupation_numbers(kpt=k, spin=s) / kd.weight_k[k]
        if not self.ppa and not self.static:
            emaxdiff = self.e_skn[:,:,self.nbands-1].max() - self.e_skn[:,:,0].min()
            assert (self.wmax > emaxdiff), 'Maximum frequency must be larger than %f' %(emaxdiff*Hartree)

        # GW kpoints init
        if (self.kpoints == None):
            self.gwnkpt = self.nikpt
            self.gwkpt_k = kd.ibz2bz_k
        else:
            self.gwnkpt = np.shape(self.kpoints)[0]
            self.gwkpt_k = self.kpoints

        # GW bands init
        if (self.bands == None):
            self.gwnband = self.nbands
            self.bands = self.gwbands_n = range(self.nbands)
        else:
            self.gwnband = np.shape(self.bands)[0]
            self.gwbands_n = self.bands

        self.alpha = 1j/(2*pi * self.vol * self.nkpt)
        
        # parallel init
        assert len(self.w_w) % self.wpar == 0
        self.wcommsize = self.wpar
        self.qcommsize = size // self.wpar
        assert self.qcommsize * self.wcommsize == size, 'wpar must be integer divisor of number of requested cores'
        if self.nqpt != 1: # parallelize over q-points
            self.wcomm, self.qcomm, self.worldcomm = set_communicator(world, rank, size, self.wpar)
            self.ncomm = serial_comm
            self.dfcomm = self.wcomm
            self.kcommsize = 1
        else: # parallelize over bands
            self.wcomm, self.ncomm, self.worldcomm = set_communicator(world, rank, size, self.wpar)
            self.qcomm = serial_comm
            if len(self.w_w) > 1:
                self.dfcomm = self.wcomm
                self.kcommsize = 1
            else:
                self.dfcomm = self.ncomm
                self.kcommsize = self.ncomm.size
        nq, self.nq_local, self.q_start, self.q_end = parallel_partition(
                                  self.nqpt, self.qcomm.rank, self.qcomm.size, reshape=False)
        nb, self.nbands_local, self.m_start, self.m_end = parallel_partition(
                                  self.nbands, self.ncomm.rank, self.ncomm.size, reshape=False)

        # print init
        self.print_gw_init()


    def get_QP_spectrum(self):
        
        self.initialize()
        
        self.printtxt("calculating Sigma")

        Sigma_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)
        dSigma_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)
        Z_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)

        t0 = time()
        t_w = 0
        t_selfenergy = 0
        for iq in range(self.q_start, self.q_end):
            if iq >= self.nqpt:
                continue
            t1 = time()
            # get screened interaction. 
            df, W_wGG = self.screened_interaction_kernel(iq, static=self.static, E0=self.E0, comm=self.dfcomm, kcommsize=self.kcommsize)
            t2 = time()
            t_w += t2 - t1

            # get self energy
            S, dS = self.get_self_energy(df, W_wGG)
            t3 = time() - t2
            t_selfenergy += t3

            Sigma_skn += S
            dSigma_skn += dS

            del df, W_wGG
            self.timing(iq, t0, self.nq_local, 'iq')

        self.printtxt('W_wGG takes %f seconds' %(t_w))
        self.printtxt('Self energy takes %f  seconds' %(t_selfenergy))

        self.qcomm.barrier()
        self.qcomm.sum(Sigma_skn)
        self.qcomm.sum(dSigma_skn)

        Z_skn = 1. / (1. - dSigma_skn)

        # exact exchange
        t0 = time()
        e_skn, vxc_skn, exx_skn = self.get_exx() # note, e_kn is different from self.e_kn
        self.printtxt('EXX takes %f seconds' %(time()-t0))

        QP_skn = e_skn + Z_skn * (Sigma_skn + exx_skn - vxc_skn)
        self.QP_skn = QP_skn

        # finish
        self.print_gw_finish(e_skn, vxc_skn, exx_skn, Sigma_skn, Z_skn, QP_skn)
        data = {
                'gwkpt_k': self.gwkpt_k,
                'gwbands_n': self.gwbands_n,
                'e_skn': e_skn,         # in Hartree
                'vxc_skn': vxc_skn,     # in Hartree
                'exx_skn': exx_skn,     # in Hartree
                'Sigma_skn': Sigma_skn, # in Hartree
                'Z_skn': Z_skn,         # dimensionless
                'QP_skn': QP_skn        # in Hartree
               }
        if rank == 0:
            pickle.dump(data, open('GW.pckl', 'w'), -1)


    def get_self_energy(self, df, W_wGG):

        Sigma_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)
        dSigma_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)

        wcomm = df.wcomm

        if self.static:
            W_wGG = np.array([W_wGG])
            if df.optical_limit:
                self.dfinvG0_wG = np.array([self.dfinvG0_G])

        # prepare optical limit for both methods
        if df.optical_limit:
            tmp_wG = np.zeros_like(self.dfinvG0_wG)
            q = np.array([0.0001,0,0])
            for jG in range(1, self.npw):
                qG = np.dot(q+self.Gvec_Gc[jG], self.bcell_cv)
                tmp_wG[:,jG] = self.dfinvG0_wG[:,jG] / np.sqrt(np.inner(qG,qG))
            tmp_wG *= 1./pi*self.vol*(6*pi**2/self.vol)**(2./3.)*self.nkpt**(1./3.)
            tmp_w = 2./pi*self.vol*(6*pi**2/self.vol)**(1./3.)*self.nkpt**(2./3.)*self.dfinvG0_wG[:,0]

        if not self.hilbert_trans: #method 1
            Wbackup_wG0 = W_wGG[:,:,0].copy()
            Wbackup_w0G = W_wGG[:,0,:].copy()

        else: #method 2, perform Hilbert transform
            nG = np.shape(W_wGG)[1]
            coords = np.zeros(wcomm.size, dtype=int)
            nG_local = nG**2 // wcomm.size
            if wcomm.rank == wcomm.size - 1:
                  nG_local = nG**2 - (wcomm.size - 1) * nG_local
            wcomm.all_gather(np.array([nG_local]), coords)
            W_Wg = SliceAlongFrequency(W_wGG, coords, wcomm)

            ng = np.shape(W_Wg)[1]
            Nw = int(self.w_w[-1] / self.dw)

            Cplus_Wg = np.zeros((Nw, ng), dtype=complex)
            Cminus_Wg = np.zeros((Nw, ng), dtype=complex)
            Cplus_wG0 = np.zeros((2, nG), dtype=complex)
            Cminus_wG0 = np.zeros((2, nG), dtype=complex)

            if df.optical_limit:
                tmp_WG = np.zeros((df.Nw, nG), dtype=complex)
                wcomm.all_gather(tmp_wG, tmp_WG)
                del tmp_wG
                tmp_W = np.zeros(df.Nw, dtype=complex)
                wcomm.all_gather(tmp_w, tmp_W)
                del tmp_w

            w1_ww = np.zeros((Nw, df.Nw), dtype=complex)
            for iw in range(Nw):
                w1 = iw * self.dw
                w1_ww[iw] = 1./(w1 + self.w_w + 1j*self.eta_w) + 1./(w1 - self.w_w + 1j*self.eta_w)
                w1_ww[iw,0] -= 1./(w1 + 1j*self.eta_w[0]) # correct w'=0
                w1_ww[iw] *= self.dw_w
            Cplus_Wg = gemmdot(w1_ww, W_Wg, beta=0.0)
            Cminus_Wg = gemmdot(w1_ww.conj(), W_Wg, beta=0.0)
            # special Hilbert transform optical limit:
            if df.optical_limit:
                Cplus_wG0 = gemmdot(w1_ww[0:2], tmp_WG, beta=0.0)
                Cminus_wG0 = gemmdot(w1_ww[0:2].conj(), tmp_WG, beta=0.0)
                Cplus_wG0[:,0] = gemmdot(w1_ww[0:2], tmp_W, beta=0.0)
                Cminus_wG0[:,0] = gemmdot(w1_ww[0:2].conj(), tmp_W, beta=0.0)

        for s in range(self.nspins):
            for i, k in enumerate(self.gwkpt_k): # k is bzk index
                if df.optical_limit:
                    kq_c = df.kd.bzk_kc[k]
                else:
                    kq_c = df.kd.bzk_kc[k] - df.q_c  # k - q

                kq = df.kd.where_is_q(kq_c, df.kd.bzk_kc)            
                assert df.kq_k[kq] == k
                ibzkpt1 = df.kd.bz2ibz_k[k]
                ibzkpt2 = df.kd.bz2ibz_k[kq]

                for j, n in enumerate(self.bands):
                    for m in range(self.m_start, self.m_end):

                        if self.f_skn[s,ibzkpt2,m] < self.ftol: #self.e_kn[s,ibzkpt2, m] > E_f :
                            sign = 1.
                        else:
                            sign = -1.

                        rho_G = df.density_matrix(m, n, kq, spin=s)

                        if not self.hilbert_trans: #method 1
                            W_wGG[:,:,0] = Wbackup_wG0
                            W_wGG[:,0,:] = Wbackup_w0G

                            if df.optical_limit:
                                if n==m:
                                    W_wGG[:,:,0] = tmp_wG
                                    W_wGG[:,0,:] = tmp_wG.conj()
                                    W_wGG[:,0,0] = tmp_w
                                else:
                                    W_wGG[:,0,0:] = 0.
                                    W_wGG[:,0:,0] = 0.

                            # w1 = w - epsilon_m,k-q
                            w1 = self.e_skn[s,ibzkpt1, n] - self.e_skn[s,ibzkpt2,m]

                            if self.ppa:
                                # analytical expression for Plasmon Pole Approximation
                                W_GG = sign * W_wGG[0] * (1./(w1 + self.wt_GG - 1j*self.eta) -
                                                          1./(w1 - self.wt_GG + 1j*self.eta))
                                W_GG -= W_wGG[0] * (1./(w1 + self.wt_GG + 1j*self.eta*sign) +
                                                    1./(w1 - self.wt_GG + 1j*self.eta*sign))
                                W_G = gemmdot(W_GG, rho_G, beta=0.0)
                                Sigma_skn[s,i,j] += np.real(gemmdot(W_G, rho_G, alpha=self.alpha, beta=0.0,trans='c'))

                                W_GG = sign * W_wGG[0] * (1./(w1 - self.wt_GG + 1j*self.eta)**2 -
                                                          1./(w1 + self.wt_GG - 1j*self.eta)**2)
                                W_GG += W_wGG[0] * (1./(w1 - self.wt_GG + 1j*self.eta*sign)**2 +
                                                    1./(w1 + self.wt_GG + 1j*self.eta*sign)**2)
                                W_G = gemmdot(W_GG, rho_G, beta=0.0)
                                dSigma_skn[s,i,j] += np.real(gemmdot(W_G, rho_G, alpha=self.alpha, beta=0.0,trans='c'))

                            elif self.static:
                                W1_GG = W_wGG[0] - np.eye(df.npw)*self.Kc_GG
                                W2_GG = W_wGG[0]
                                if df.optical_limit:
                                    if n==m:
                                        W1_GG[0,0] = tmp_w[0] - 2./pi*self.vol*(6*pi**2/self.vol)**(1./3.)*self.nkpt**(2./3.)
                                    else:
                                        W1_GG[0,0:] = 0.
                                        W1_GG[0:,0] = 0.
                                # perform W_GG * np.outer(rho_G.conj(), rho_G).sum(GG)
                                W_G = gemmdot(W1_GG, rho_G, beta=0.0) # Coulomb Hole
                                Sigma_skn[s,i,j] += np.real(gemmdot(W_G, rho_G, alpha=self.alpha*pi/1j, beta=0.0,trans='c'))
                                if sign == -1:
                                    W_G = gemmdot(W2_GG, rho_G, beta=0.0) # Screened Exchange
                                    Sigma_skn[s,i,j] -= np.real(gemmdot(W_G, rho_G, alpha=2*self.alpha*pi/1j, beta=0.0,trans='c'))
                                del W1_GG, W2_GG, W_G, rho_G

                            else:
                                # perform W_wGG * np.outer(rho_G.conj(), rho_G).sum(GG)
                                W_wG = gemmdot(W_wGG, rho_G, beta=0.0)
                                C_wlocal = gemmdot(W_wG, rho_G, alpha=self.alpha, beta=0.0,trans='c')
                                del W_wG, rho_G

                                C_w = np.zeros(df.Nw, dtype=complex)
                                wcomm.all_gather(C_wlocal, C_w)
                                del C_wlocal

                                # calculate self energy
                                w1_w = 1./(w1 - self.w_w + 1j*self.eta_w*sign) + 1./(w1 + self.w_w + 1j*self.eta_w*sign)
                                w1_w[0] -= 1./(w1 + 1j*self.eta_w[0]*sign) # correct w'=0
                                w1_w *= self.dw_w
                                Sigma_skn[s,i,j] += np.real(gemmdot(C_w, w1_w, beta=0.0))

                                # calculate derivate of self energy with respect to w
                                w1_w = 1./(w1 - self.w_w + 1j*self.eta_w*sign)**2 + 1./(w1 + self.w_w + 1j*self.eta_w*sign)**2
                                w1_w[0] -= 1./(w1 + 1j*self.eta_w[0]*sign)**2 # correct w'=0
                                w1_w *= self.dw_w
                                dSigma_skn[s,i,j] -= np.real(gemmdot(C_w, w1_w, beta=0.0))

                        else: #method 2
                            if not np.abs(self.e_skn[s,ibzkpt2,m] - self.e_skn[s,ibzkpt1,n]) < 1e-10:
                                sign *= np.sign(self.e_skn[s,ibzkpt1,n] - self.e_skn[s,ibzkpt2,m])

                            # find points on frequency grid
                            w0 = self.e_skn[s,ibzkpt1,n] - self.e_skn[s,ibzkpt2,m]
                            w0_id = np.abs(int(w0 / self.dw))
                            w1 = w0_id * self.dw
                            w2 = (w0_id + 1) * self.dw

                            # choose plus or minus, treat optical limit:
                            if sign == 1:
                                C_Wg = Cplus_Wg[w0_id:w0_id+2] # only two grid points needed for each w0
                            if sign == -1:
                                C_Wg = Cminus_Wg[w0_id:w0_id+2] # only two grid points needed for each w0

                            C_wGG = GatherOrbitals(C_Wg, coords, wcomm)
                            del C_Wg

                            if df.optical_limit:
                                if n==m:
                                    if sign == 1:
                                        C_wGG[:,0,:] = Cminus_wG0.conj()
                                        C_wGG[:,:,0] = Cplus_wG0
                                    if sign == -1:
                                        C_wGG[:,0,:] = Cplus_wG0.conj()
                                        C_wGG[:,:,0] = Cminus_wG0
                                else:
                                    C_wGG[:,0,0:] = 0.
                                    C_wGG[:,0:,0] = 0.

                            # special treat of w0 = 0 (degenerate states):
                            if w0_id == 0:
                                Cplustmp_GG = GatherOrbitals(Cplus_Wg[1], coords, wcomm)
                                Cminustmp_GG = GatherOrbitals(Cminus_Wg[1], coords, wcomm)
                                if df.optical_limit:
                                    if n==m:
                                        Cplustmp_GG[0,:] = Cminus_wG0.conj()[1]
                                        Cplustmp_GG[:,0] = Cplus_wG0[1]
                                        Cminustmp_GG[0,:] = Cplus_wG0.conj()[1]
                                        Cminustmp_GG[:,0] = Cminus_wG0[1]
                                    else:
                                        Cplustmp_GG[0,:] = 0.
                                        Cplustmp_GG[:,0] = 0.
                                        Cminustmp_GG[0,:] = 0.
                                        Cminustmp_GG[:,0] = 0.

                            # perform C_wGG * np.outer(rho_G.conj(), rho_G).sum(GG)

                            if w0_id == 0:
                                Sw0_G = gemmdot(C_wGG[0], rho_G, beta=0.0)
                                Sw0 = np.real(gemmdot(Sw0_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))
                                Sw1_G = gemmdot(Cplustmp_GG, rho_G, beta=0.0)
                                Sw1 = np.real(gemmdot(Sw1_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))
                                Sw2_G = gemmdot(Cminustmp_GG, rho_G, beta=0.0)
                                Sw2 = np.real(gemmdot(Sw2_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))

                                Sigma_skn[s,i,j] += Sw0
                                dSigma_skn[s,i,j] += (Sw1 + Sw2)/(2*self.dw)

                            else:                        
                                Sw1_G = gemmdot(C_wGG[0], rho_G, beta=0.0)
                                Sw1 = np.real(gemmdot(Sw1_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))
                                Sw2_G = gemmdot(C_wGG[1], rho_G, beta=0.0)
                                Sw2 = np.real(gemmdot(Sw2_G, rho_G, alpha=self.alpha, beta=0.0, trans='c'))

                                Sw0 = (w2-np.abs(w0))/self.dw * Sw1 + (np.abs(w0)-w1)/self.dw * Sw2
                                Sigma_skn[s,i,j] += np.sign(self.e_skn[s,ibzkpt1,n] - self.e_skn[s,ibzkpt2,m]) * Sw0
                                dSigma_skn[s,i,j] += (Sw2 - Sw1)/self.dw

        self.ncomm.barrier()
        self.ncomm.sum(Sigma_skn)
        self.ncomm.sum(dSigma_skn)

        return Sigma_skn, dSigma_skn 


    def get_exx(self):

        if self.exxfile:
            self.printtxt("reading Exact exchange and E_XC from file")

            data = pickle.load(open(self.exxfile))
            e_skn = data['e_skn'] # in Hartree
            vxc_skn = data['vxc_skn'] # in Hartree
            if not self.static:
                exx_skn = data['exx_skn'] # in Hartree
            else:
                exx_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)
            gwkpt_k = data['gwkpt_k']
            gwbands_n = data['gwbands_n']
            assert (gwkpt_k == self.gwkpt_k).all(), 'exxfile inconsistent with input parameters'
            assert (gwbands_n == self.gwbands_n).all(), 'exxfile inconsistent with input parameters'

        else:
            self.printtxt("calculating Exact exchange and E_XC")

            calc = GPAW(self.file, communicator=world, parallel={'domain':1}, txt=None)
            v_xc = vxc(calc)

            if not self.static:
                if isinstance(calc.wfs, PWWaveFunctions): # planewave mode
                    from gpaw.xc.hybridg import HybridXC
                    exx = HybridXC('EXX', alpha=5.0, bandstructure=True, bands=self.bands)
                else:                                     # grid mode
                    from gpaw.xc.hybridk import HybridXC
                    exx = HybridXC('EXX', alpha=5.0, ecut=self.ecut.max(), bands=self.bands)
                calc.get_xc_difference(exx)

            e_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)
            vxc_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)
            exx_skn = np.zeros((self.nspins, self.gwnkpt, self.gwnband), dtype=float)

            for s in range(self.nspins):
                for i, k in enumerate(self.gwkpt_k):
                    ik = self.kd.bz2ibz_k[k]
                    for j, n in enumerate(self.gwbands_n):
                        e_skn[s][i][j] = calc.get_eigenvalues(kpt=ik, spin=s)[n] / Hartree
                        vxc_skn[s][i][j] = v_xc[s][ik][n] / Hartree
                        if not self.static:
                            exx_skn[s][i][j] = exx.exx_skn[s][ik][n]

        return e_skn, vxc_skn, exx_skn


    def print_gw_init(self):

        self.printtxt("Number of IBZ k-points       : %d" %(self.kd.nibzkpts))
        self.printtxt("Number of spins              : %d" %(self.nspins))
        self.printtxt('')
        if self.ppa:
            self.printtxt("Use Plasmon Pole Approximation")
            self.printtxt("imaginary frequency (eV)     : %.2f" %(self.E0))
        elif self.static:
            self.printtxt("Use static COHSEX")
        else:
            self.printtxt("Linear frequency grid (eV)   : %.2f - %.2f in %.2f" %(self.wmin*Hartree, self.wcut, self.dw*Hartree))
            self.printtxt("Maximum frequency (eV)       : %.2f" %(self.wmax*Hartree))
            self.printtxt("Number of frequency points   : %d" %(self.Nw))
        self.printtxt('')
        self.printtxt('Calculate matrix elements for k = :')
        for k in self.gwkpt_k:
            self.printtxt(self.kd.bzk_kc[k])
        self.printtxt('')
        self.printtxt('Calculate matrix elements for n = %s' %(self.gwbands_n))


    def print_gw_finish(self, e_skn, vxc_skn, exx_skn, Sigma_skn, Z_skn, QP_skn):

        self.printtxt("------------------------------------------------")
        self.printtxt("LDA eigenvalues are (eV): ")
        self.printtxt("%s \n" %(e_skn*Hartree))
        self.printtxt("LDA exchange-correlation contributions are (eV): ")
        self.printtxt("%s \n" %(vxc_skn*Hartree))
        if not self.static:
            self.printtxt("Exact exchange contributions are (eV): ")
            self.printtxt("%s \n" %(exx_skn*Hartree))
        self.printtxt("Self energy contributions are (eV):")
        self.printtxt("%s \n" %(Sigma_skn*Hartree))
        if not self.static:
            self.printtxt("Renormalization factors are:")
            self.printtxt("%s \n" %(Z_skn))

        totaltime = round(time() - self.starttime)
        self.printtxt("GW calculation finished in %s " %(timedelta(seconds=totaltime)))
        self.printtxt("------------------------------------------------")
        self.printtxt("Quasi-particle energies are (eV): ")
        self.printtxt(QP_skn*Hartree)
