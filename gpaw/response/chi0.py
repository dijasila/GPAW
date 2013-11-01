"""Todo:

* metals
* parallelize over spin and bands
* optical limit
"""

from math import pi

import numpy as np
from ase.units import Hartree

from gpaw.mpi import world
from gpaw.utilities.blas import gemm, rk
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.math_func import two_phi_planewave_integrals


class Chi0:
    def __init__(self, calc, omega_w, ecut=50 / Hartree, hilbert=False,
                 eta=0.2 / Hartree, blocksize=50):
        self.calc = calc
        self.omega_w = omega_w
        self.ecut = ecut
        self.hilbert = hilbert
        self.eta = eta
        self.blocksize = blocksize

        self.spos_ac = calc.atoms.get_scaled_positions()
        
        self.mynk = None
        self.eps_nk = None
        self.ut_knR = None
        self.P_kani = None
        
        self.initialize_occupied_states()
        
    def initialize_occupied_states(self):
        wfs = self.calc.wfs
        kd = wfs.kd
        
        vol = abs(np.linalg.det(wfs.gd.cell_cv))
        self.prefactor = 2 / vol / kd.nbzkpts
        
        nocc = wfs.nvalence // 2
        self.mynk = kd.nbzkpts // world.size
        
        self.ut_knR = wfs.gd.empty((self.mynk, nocc), complex)
        self.eps_kn = np.empty((self.mynk, nocc))
        self.P_kani = []
        self.shift_kc = []
        for k in range(self.mynk):
            K = k + world.rank * self.mynk
            ut_nG, eps_n, P_ani, shift_c = self.get_k_point(K, 0, nocc)
            self.ut_knR[k] = ut_nG
            self.eps_kn[k] = eps_n
            self.P_kani.append(P_ani)
            self.shift_kc.append(shift_c)
            
    def get_k_point(self, K, n1, n2):
        wfs = self.calc.wfs
        
        T, T_a, shift_c = self.construct_symmetry_operators(K)
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[ik]
        
        psit_nG = kpt.psit_nG
        ut_nR = wfs.gd.empty(n2 - n1, complex)
        for n in range(n1, n2):
            ut_nR[n - n1] = T(wfs.pd.ifft(psit_nG[n], ik))

        eps_n = kpt.eps_n[n1:n2]
        
        P_ani = []
        for (b, T_ii, time_reversal) in T_a:
            P_ni = np.dot(kpt.P_ani[b][n1:n2], T_ii)
            if time_reversal:
                P_ni = P_ni.conj()
            P_ani.append(P_ni)
        
        return ut_nR, eps_n, P_ani, shift_c
    
    def calculate(self, q_c):
        wfs = self.calc.wfs

        if self.eta == 0:
            update = self.update_hermetian_chi0
        elif self.hilbert:
            update = self.update_hilbert
        else:
            update = self.update_chi0
            
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)
        nG = pd.ngmax
        chi0_wGG = np.zeros((len(self.omega_w), nG, nG), complex)
        
        self.initialize_paw_corrections(pd)
        
        nocc = wfs.nvalence // 2
        nunocc = wfs.bd.nbands - nocc

        for k in range(self.mynk):
            P_ani = self.P_kani[k]
            K = k + world.rank * self.mynk
            K2 = wfs.kd.find_k_plus_q(q_c, [K])[0]
            ut_mR, eps_m, P_ami, shift_c = self.get_k_point(K2, nocc,
                                                            nocc + nunocc)
            Q_G = self.get_fft_indices(K, K2, q_c, pd,
                                       self.shift_kc[k] - shift_c)
            for n in range(nocc):
                eps = self.eps_kn[k, n]
                utcc_R = self.ut_knR[k, n].conj()
                C_aGi = [np.dot(Q_Gii, P_ni[n].conj())
                         for Q_Gii, P_ni in zip(self.Q_aGii, P_ani)]
                for m1 in range(0, nunocc, self.blocksize):
                    m2 = min(m1 + self.blocksize, nunocc)
                    n_mG = self.calculate_pair_densities(utcc_R, C_aGi,
                                                         ut_mR, P_ami,
                                                         m1, m2, pd, Q_G)
                    omega_m = eps - eps_m[m1:m2]
                    update(n_mG, omega_m, chi0_wGG)
                    
        world.sum(chi0_wGG)
        
        if self.eta == 0:
            il = np.tril_indices(nG, -1)
            iu = il[::-1]
            for chi0_GG in chi0_wGG:
                chi0_GG[il] = chi0_GG[iu]
                
        if self.hilbert:
            for G in range(nG):
                chi0_wGG[:, :, G] = np.dot(A_ww, chi0_wGG[:, :, G])
        
        return chi0_wGG, pd
        
    def update_chi0(self, n_mG, omega_m, chi0_wGG):
        for w, omega in enumerate(self.omega_w):
            x_m = (1.0 / (omega + omega_m + 1j * self.eta) -
                   1.0 / (omega - omega_m + 1j * self.eta))
            x_mG = n_mG * x_m[:, np.newaxis]
            gemm(self.prefactor, n_mG.conj(), np.ascontiguousarray(x_mG.T),
                 1.0, chi0_wGG[w])

    def update_hermitian_chi0(self, n_mG, omega_m, chi0_wGG):
        for w, omega in enumerate(self.omega_w):
            x_m = (omega_m / (omega**2 + omega_m**2))**0.5
            x_mG = n_mG * x_m[:, np.newaxis]
            rk(self.prefactor, x_mG, 1.0, chi0_wGG[w])

    def calculate_pair_densities(self, utcc_R, C_aGi, ut_mR, P_ami,
                                 m1, m2, pd, Q_G):
        dv = pd.gd.dv
        n_mG = pd.empty(m2 - m1)
        for m in range(m1, m2):
            n_R = utcc_R * ut_mR[m]
            pd.tmp_R[:] = n_R
            pd.fftplan.execute()
            n_mG[m - m1] = pd.tmp_Q.ravel()[Q_G] * dv
        
        # PAW corrections:
        for C_Gi, P_mi in zip(C_aGi, P_ami):
            gemm(1.0, C_Gi, P_mi[m1:m2], 1.0, n_mG, 't')
            
        return n_mG

    def get_fft_indices(self, K, K2, q_c, pd, shift0_c):
        kd = self.calc.wfs.kd
        Q_G = pd.Q_qG[0]
        shift_c = (shift0_c +
                   (q_c - kd.bzk_kc[K2] + kd.bzk_kc[K]).round().astype(int))
        if shift_c.any():
            q_cG = np.unravel_index(Q_G, pd.gd.N_c)
            q_cG = [q_G + shift for q_G, shift in zip(q_cG, shift_c)]
            Q_G = np.ravel_multi_index(q_cG, pd.gd.N_c, 'wrap')
        return Q_G
        
    def construct_symmetry_operators(self, K):
        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]
        
        shift_c = np.dot(U_cc, ik_c) - k_c * (1 - 2 * time_reversal)
        assert (shift_c.round() == shift_c).all()
        shift_c = shift_c.round().astype(int)
        
        if (U_cc == np.eye(3)).all():
            T = lambda f_R: f_R
        else:
            N_c = self.calc.wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            T = lambda f_R: f_R.ravel()[i].reshape(N_c)

        if time_reversal:
            T0 = T
            T = lambda f_R: T0(f_R).conj()
            shift_c *= -1
        
        T_a = []
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            T_a.append((b, U_ii, time_reversal))

        return T, T_a, shift_c

    def initialize_paw_corrections(self, pd):
        wfs = self.calc.wfs
        q_v = pd.K_qv[0]
        G_Gv = pd.G_Qv[pd.Q_qG[0]] + q_v
        pos_av = np.dot(self.spos_ac, pd.gd.cell_cv)
            
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
            Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
            ni = atomdata.ni
            Q_xGii[id] = Q_Gii.reshape((-1, ni, ni))
        
        self.Q_aGii = []
        for a, atomdata in enumerate(wfs.setups):
            Q_Gii = Q_xGii[wfs.setups.id_a[a]]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            self.Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
