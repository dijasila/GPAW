
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


import matplotlib.pyplot as plt
from numpy.matlib import repmat

class Full4C(RIAlgorithm):
    def __init__(self, exx_fraction=None, screening_omega=None):
        RIAlgorithm.__init__(self, 'Full4C debug', exx_fraction, screening_omega)
        self.K_kkMMMM = {}
        self.only_ghat = False
        self.no_ghat = False
        self.only_ghat_aux_interaction = False

    def set_positions(self, spos_ac):
        RIAlgorithm.set_positions(self, spos_ac)
        self.spos_ac = spos_ac

    def calculate_exchange_per_kpt_pair(self, kpt1, k_c, rho1_MM, kpt2, krho_c, rho2_MM):
        K_MMMM = self.get_K_MMMM(kpt1, k_c, kpt2, krho_c)
        V_MM = -(self.nspins/2.)*self.exx_fraction * np.einsum('ikjl,kl', K_MMMM, rho2_MM)
        E = 0.5 * np.einsum('ij,ij', rho1_MM, V_MM)
        #print(kpt2.q, kpt2.f_n, kpt2.eps_n)
        return E.real, V_MM

    def get_K_MMMM(self, kpt1, k1_c, kpt2, k2_c):
        first = True
        # Caching for precalculated results
        #print('Requesting K_MMMM for kpt pair ', k1_c, k2_c)
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
        print(K_MMMM,'K_MMMM')
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
        pd12 = PWDescriptor(None, finegd, complex, kd=qd) #10 Ha =  270 eV cutoff
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
            #if first:
            #    first = False
            #    plt.plot(slice_G.real, label=str(kpt1.q)+'r')
            #    plt.plot(slice_G.imag)
            #    plt.show()

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
            if self.only_ghat:
                print('Only compesation charges')
                rho_xG[:] = 0.0
            if not self.no_ghat:
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
        print(K_MMMM,'K_MMMM')
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
        #print('Allocating ', 2*rho_pg.itemsize / 1024.0**2,' MB')

        dtype = self.wfs.basis_functions.dtype
        # Put wave functions of the two basis functions to the grid
        phit1_MG = gd.zeros(nao, dtype=dtype)
        self.wfs.basis_functions.lcao_to_grid(np.eye(nao, dtype=dtype), phit1_MG, kpt1.q)
       
        assert np.linalg.norm(phit1_MG.ravel().imag)<1e-10
        phit1_MG = phit1_MG.real.copy()

        for p1, (M1, M2) in enumerate(pairs_p):
            if p1 % 100 == 0:
                print('Potentials for pair %d/%d' % (p1, len(pairs_p)))
            # Construct the product of two basis functions
            
            rhot_G = phit1_MG[M1] * phit1_MG[M2]
            rhot_g = finegd.zeros()
            interpolator.apply(rhot_G, rhot_g)

            print('real space norm', finegd.integrate(rhot_g))

            # Add compensation charges in reciprocal space
            Q_aL = {}
            D_ap = {}
            for a in self.wfs.P_aqMi:
                P1_i = self.wfs.P_aqMi[a][kpt1.q][M1]
                P2_i = self.wfs.P_aqMi[a][kpt2.q][M2]
                D_ii = np.outer(P1_i, P2_i.conjugate())
                D_p = pack(D_ii)
                D_ap[a] = D_p
                Q_L = np.dot(D_p, self.density.setups[a].Delta_pL)
                assert np.linalg.norm(Q_L.imag)<1e-10
                Q_aL[a] = Q_L.real.copy()
            
            if self.only_ghat:
                print('Only compesation charges')
                rhot_g[:] = 0.0

            if self.only_ghat_aux_interaction:
                print('only ghat_aux_interaction')
                rhot2_g = np.zeros_like(rhot_g)
                self.density.ghat.add(rhot2_g, Q_aL)

            elif not self.no_ghat:
                self.density.ghat.add(rhot_g, Q_aL)

            V_g = finegd.zeros()
            self.hamiltonian.poisson.solve(V_g, rhot_g, charge=None)

            if self.only_ghat_aux_interaction:
                rho_pg[p1][:] = 2*rhot2_g
            else:
                rho_pg[p1][:] = rhot_g
            V_pg[p1][:] = V_g

        K_pp = ( finegd.integrate(rho_pg, V_pg) + finegd.integrate(V_pg, rho_pg) ) / 2
        for p1, (M1, M2) in enumerate(pairs_p):
            for p2, (M3, M4) in enumerate(pairs_p):
                K = K_pp[p1, p2]
                K_MMMM[M1,M2,M3,M4] = K

        return K_MMMM

    def get_description(self):
        return 'Debug evaluation of full 4 center matrix elements'
