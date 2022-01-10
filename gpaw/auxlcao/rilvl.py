from typing import Tuple, Dict
import numpy as np
from gpaw.auxlcao.algorithm import RIAlgorithm
from gpaw.hybrids.coulomb import ShortRangeCoulomb
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.utilities import pack, unpack2
from gpaw.pw.lfc import PWLFC
from gpaw.transformers import Transformer
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC

from gpaw.auxlcao.multipole import calculate_W_qLL
from gpaw.auxlcao.procedures import calculate_V_qAA

import matplotlib.pyplot as plt
from numpy.matlib import repmat

class RIBasisMaker:
    def __init__(self, setup):
        self.setup = setup


class RILVL(RIAlgorithm):
    def __init__(self, exx_fraction=None, screening_omega=None, lcomp=2):
        RIAlgorithm.__init__(self, 'RI-LVL', exx_fraction, screening_omega)
        self.lcomp = lcomp
        assert self.lcomp == 2
        self.K_kkMMMM={}

    def prepare_setups(self):
        RIAlgorithm.prepare_setups(self)

    def set_positions(self, spos_ac):
        RIAlgorithm.set_positions(self, spos_ac)

        kd = self.wfs.kd
        self.bzq_qc = bzq_qc = kd.get_bz_q_points()
        print('Number of q-points: ', len(bzq_qc))

        with self.timer('RI-V: calculate W_qLL'):
             self.W_qLL = calculate_W_qLL(self.density.setups,\
                                          self.hamiltonian.gd.cell_cv,
                                          spos_ac,
                                          self.hamiltonian.gd.pbc_c,
                                          bzq_qc,
                                          self.wfs.dtype,
                                          self.lcomp, omega=self.screening_omega)

       
        for q, bzq_c in enumerate(bzq_qc):
            with self.timer('RI-V: calculate V_AA'):
                self.V_qAA = calculate_V_qAA(auxt_aj, M_aj, self.W_LL, self.lmaxcomp)
                assert not np.isnan(self.V_qAA).any()

            with self.timer('RI-V: calculate S_AA'):
                self.S_qAA = calculate_S_qAA(self.matrix_elements)
                assert not np.isnan(self.S_qAA).any()

            with self.timer('RI-V: calculate M_AA'):
                self.M_qAA = calculate_M_AA(self.matrix_elements, auxt_aj, M_aj, self.lmaxcomp)
                self.W_qAA = self.V_qAA + self.S_qAA + self.M_qAA + self.M_qAA.T
                assert not np.isnan(self.M_qAA).any()

        for pair in self.kpt_pairs:
            with self.timer('RI-V: Calculate P_AMM'):
                self.P_kAMM[pair] = calculate_P_AMM(self.matrix_elements, self.W_AA)
                assert not np.isnan(self.P_AMM).any()

        with self.timer('RI-V: Calculate P_LMM'):
            self.P_LMM = calculate_P_LMM(self.matrix_elements, self.wfs.setups, self.wfs.atomic_correction)
            assert not np.isnan(self.P_LMM).any()

        with self.timer('RI-V: Calculate W_AL'):
            self.W_AL = calculate_W_AL(self.matrix_elements, auxt_aj, M_aj, self.W_LL)
            assert not np.isnan(self.W_AL).any()

        with self.timer('RI-V: Calculate WP_AMM'):

            self.WP_AMM = np.einsum('AB,Bij',self.W_AA, self.P_AMM, optimize=True)
            self.WP_AMM += np.einsum('AB,Bij',self.W_AL, self.P_LMM, optimize=True)

        with self.timer('RI-V: Calculate WP_LMM'):
            self.WP_LMM = np.einsum('BA,Bij',self.W_AL, self.P_AMM, optimize=True)
            self.WP_LMM += np.einsum('AB,Bij',self.W_LL, self.P_LMM, optimize=True)


    def calculate_exchange_per_kpt_pair(self, kpt1, k_c, rho1_MM, kpt2, krho_c, rho2_MM):
        kpt_pair = (kpt1.q, kpt2.q)
        with self.timer('RI-V: 1st contraction AMM MM'):
            WP_AMM_RHO_MM = np.einsum('Ajl,kl',
                                        self.WP_kkAMM[kpt_pair],
                                        rho2_MM, optimize=True)

        with self.timer('RI-V: 2nd contraction AMM AMM'):
            F_MM = np.einsum('Aik,Ajk',
                              self.P_kkAMM[kpt_pair],
                              WP_AMM_RHO_MM,
                              optimize=True)
            WP_AMM_RHO_MM = None

        with self.timer('RI-V: 1st contraction LMM MM'):
            WP_LMM_RHO_MM = np.einsum('Ajl,kl',
                                       self.WP_kkLMM[kpt_pair],
                                       rho2_MM, optimize=True)

        with self.timer('RI-V: 2nd contraction LMM LMM'):
            F_MM += np.einsum('Aik,Ajk',
                              self.P_kkLMM[kpt_pair],
                              WP_LMM_RHO_MM,
                              optimize=True)
            WP_LMM_RHO_MM = None

        F_MM *= -0.5*self.exx_fraction
        evv = -0.5 * 0.5 * self.exx_fraction * np.einsum('ij,ij', F_MM, rho1_MM, optimize=True)
        return evv.real, F_MM

    def get_K_MMMM(self, kpt1, k1_c, kpt2, k2_c):
        k12_c = str((k1_c, k2_c))
        if k12_c in self.K_kkMMMM:
            return self.K_kkMMMM[k12_c]

        if self.screening_omega != 0.0 or np.all(self.density.gd.pbc_c):
            K_MMMM = self.get_K_MMMM_pbc(kpt1, k1_c, kpt2, k2_c)
        elif self.screening_omega == 0.0 and not np.any(self.density.gd.pbc_c):
            K_MMMM = self.get_K_MMMM_finite(kpt1, k1_c, kpt2, k2_c)
        else:
            raise NotImplementedError    

        self.K_kkMMMM[k12_c] = K_MMMM
        return K_MMMM

    def get_K_MMMM_pbc(self, kpt1, k1_c, kpt2, k2_c):
        wfs = self.wfs
        density = self.density

        q_c = (k1_c - k2_c)
        print(q_c,'q_c',k1_c,k2_c)

        finegd = GridDescriptor(self.density.finegd.N_c, 
                                self.density.finegd.cell_cv,
                                pbc_c=True)

        # Create descriptors for the charge density (with symmetry q_c)
        if self.screening_omega != 0.0:
            coulomb = ShortRangeCoulomb(self.screening_omega)
        else:
            coulomb = WSTC(finegd.cell_cv, np.array([3,3,3]))


        gd = self.density.gd

        interpolator = Transformer(gd, finegd, 1, complex)
        
        #qd = KPointDescriptor([0*q_c])
        qd = KPointDescriptor([-q_c])
        pd12 = PWDescriptor(10.0, finegd, complex, kd=qd) #10 Ha =  270 eV cutoff
        v_G = coulomb.get_potential(pd12)

        # Exchange compensation charges
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)

        nao = self.wfs.setups.nao
        K_MMMM = np.zeros( (nao, nao, nao, nao ), dtype=complex )
        #print('Allocating ', K_MMMM.itemsize / 1024.0**2,' MB')

        #gd = self.density.gd

        pairs_p = []
        for M1 in range(nao):
            for M2 in range(nao):
                pairs_p.append((M1,M2))

        # Since this is debug, do not care about memory use
        rho_pG = pd12.zeros( len(pairs_p), dtype=complex, q=0)
        V_pG = pd12.zeros( len(pairs_p), dtype=complex, q=0 )
        #print('Allocating ', 2*rho_pG.itemsize / 1024.0**2,' MB')

        # Put wave functions of the two basis functions to the grid
        phit1_MG = gd.zeros(nao, dtype=complex)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao, dtype=complex), phit1_MG, kpt1.q)

        phit2_MG = gd.zeros(nao, dtype=complex)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao, dtype=complex), phit2_MG, kpt2.q)

        for p1, (M1, M2) in enumerate(pairs_p):
            if p1 % 100 == 0:
                print('Potentials for pair %d/%d' % (p1, len(pairs_p)))
            rhot_xG = self.density.gd.zeros((1,), dtype=complex) 
            # Construct the product of two basis functions
            rhot_xG[0][:] = phit1_MG[M1].conjugate() * phit2_MG[M2] * density.gd.plane_wave(q_c)

            slice_G = rhot_xG[0,0,0,:]
            slice_G = np.concatenate((slice_G, slice_G))

            rhot_xg = finegd.zeros((1,), dtype=complex)
            interpolator.apply(rhot_xG, rhot_xg, np.ones((3, 2), complex))

            #if 1:
            #    N = np.linalg.norm(rhot_xg - ghat.pd.ifft(ghat.pd.fft(rhot_xg)))
            #    print('Norms', N)
            #    assert N<1e-7

            #print(self.density.gd.integrate(rhot_xG),'Real space norm')

            rho_xG = pd12.fft(rhot_xg) # Fourier transform the Bloch function product
            rho_xG = rho_xG.reshape( (1,) + rho_xG.shape)
            # Add compensation charges in reciprocal space
            Q_aL = {}
            D_ap = {}
            for a in self.wfs.P_aqMi:
                P1_i = self.wfs.P_aqMi[a][kpt1.q][M1]
                P2_i = self.wfs.P_aqMi[a][kpt2.q][M2]
                D_ii = np.outer(P1_i, P2_i.conjugate())
                D_p = pack(D_ii)
                D_ap[a] = D_p
                tmp = np.dot(D_p, self.density.setups[a].Delta_pL)
                Q_aL[a] = tmp.reshape((1,) + tmp.shape)
            ghat.add(rho_xG, Q_aL)

            Vrho_G = rho_xG[0] * v_G

            pot = pd12.ifft(Vrho_G)

            rho_pG[p1][:] = rho_xG[0]
            V_pG[p1][:] = Vrho_G

        K_pp = pd12.integrate(rho_pG, V_pG)
        for p1, (M1, M2) in enumerate(pairs_p):
            for p2, (M3, M4) in enumerate(pairs_p):
                K = K_pp[p1, p2]
                K_MMMM[M1,M2,M3,M4] = K

        return K_MMMM

    def get_K_MMMM_finite(self, kpt1, k1_c, kpt2, k2_c):
        assert kpt1.q == 0
        assert kpt2.q == 0
        wfs = self.wfs
        density = self.density

        q_c = (k1_c - k2_c)
        assert np.all(q_c == 0.0)

        finegd = self.density.finegd 

        # Create descriptors for the charge density (with symmetry q_c)
        if self.screening_omega != 0.0:
            raise NotImplementedError

        gd = self.density.gd
        interpolator = self.density.interpolator
        restrictor = self.hamiltonian.restrictor
        
        nao = self.wfs.setups.nao
        K_MMMM = np.zeros( (nao, nao, nao, nao )  )

        pairs_p = []
        for M1 in range(nao):
            for M2 in range(nao):
                pairs_p.append((M1,M2))

        # Since this is debug, do not care about memory use
        rho_pg = finegd.zeros( len(pairs_p) )
        V_pg = finegd.zeros( len(pairs_p) )
        print('Allocating ', 2*rho_pg.itemsize / 1024.0**2,' MB')

        # Put wave functions of the two basis functions to the grid
        phit1_MG = gd.zeros(nao)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao), phit1_MG, kpt1.q)

        for p1, (M1, M2) in enumerate(pairs_p):
            if p1 % 100 == 0:
                print('Potentials for pair %d/%d' % (p1, len(pairs_p)))
            # Construct the product of two basis functions
            rhot_G = phit1_MG[M1] * phit1_MG[M2]

            rhot_g = finegd.zeros()
            interpolator.apply(rhot_G, rhot_g)

            # Add compensation charges in reciprocal space
            Q_aL = {}
            D_ap = {}
            for a in self.wfs.P_aqMi:
                P1_i = self.wfs.P_aqMi[a][kpt1.q][M1]
                P2_i = self.wfs.P_aqMi[a][kpt2.q][M2]
                D_ii = np.outer(P1_i, P2_i.conjugate())
                D_p = pack(D_ii)
                D_ap[a] = D_p
                Q_aL[a] = np.dot(D_p, self.density.setups[a].Delta_pL)

            self.density.ghat.add(rhot_g, Q_aL)

            V_g = finegd.zeros()
            self.hamiltonian.poisson.solve(V_g, rhot_g, charge=None)

            rho_pg[p1][:] = rhot_g
            V_pg[p1][:] = V_g

        K_pp = finegd.integrate(rho_pg, V_pg)
        for p1, (M1, M2) in enumerate(pairs_p):
            for p2, (M3, M4) in enumerate(pairs_p):
                K = K_pp[p1, p2]
                K_MMMM[M1,M2,M3,M4] = K

        return K_MMMM

    def get_description(self):
        return 'RI-LVL'
