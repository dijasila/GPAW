from typing import Tuple, Dict
import numpy as np
from gpaw.auxlcao.algorithm import RIAlgorithm
from gpaw.hybrids.coulomb import ShortRangeCoulomb
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor


def charge_density(wfs, density, rho_MM):
    Q_aL = {}
    D_ap = {}
    q = 0
    for a in wfs.P_aqMi:
        P_Mi = wfs.P_aqMi[a][0]
        D_ii = np.dot(P_Mi.T, np.dot(rho_MM, P_Mi))
        D_p = pack(D_ii)
        D_ap[a] = D_p
        Q_aL[a] = np.dot(D_p, density.setups[a].Delta_pL).real.copy()
    rho_G = density.gd.zeros()
    wfs.basis_functions.construct_density(rho_MM, rho_G, q)
    rho_g = density.finegd.zeros()
    density.interpolator.apply(rho_G, rho_g)
    density.ghat.add(rho_g, Q_aL)
    return rho_g, D_ap


class Full4C(RIAlgorithm):
    def __init__(self, exx_fraction=None, screening_omega=None):
        RIAlgorithm.__init__(self, 'Full4C debug', exx_fraction, screening_omega)
        self.K_kkMMMM = {}

    def set_positions(self, spos_ac):
        RIAlgorithm.set_positions(self, spos_ac)
        self.spos_ac = spos_ac

    def calculate_exchange_per_kpt_pair(self, k_c, krho_c, rho_MM):
        K_MMMM = self.get_K_MMMM(k_c, krho_c)
        return -1/2*self.exx_fraction * np.einsum('ikjl,kl', K_MMMM, rho_MM)

    def get_K_MMMM(self, k1_c, k2_c):
        nao = self.wfs.setups.nao
        rho_MM = np.zeros( (nao, nao) )
        print('Requesting K_MMMM for kpt pair ', k1_c, k2_c)
        k12_c = str((k1_c, k2_c))
        if k12_c in self.K_kkMMMM:
            return self.K_kkMMMM[k12_c]

        K_MMMM = np.zeros( (nao, nao, nao, nao ) )


        q_c = k1_c - k2_c # Sign?
        if self.screening_omega != 0.0:
            coulomb = ShortRangeCoulomb(self.screening_omega)
            finegd = GridDescriptor(self.density.finegd.N_c, 
                                    self.density.finegd.cell_cv,
                                    pbc_c=True)

            qd = KPointDescriptor([np.array([q_c])  ])
            pd12 = PWDescriptor(None, finegd, np.float, kd=qd)
            v_G = coulomb.get_potential(pd12)


        pairs_p = []
        for M1 in range(nao):
            for M2 in range(M1, nao):
                pairs_p.append((M1,M2))

        psit_nG = gd.zeros(nao)
        basis_functions.lcao_to_grid(np.eye(nao), kpt.psit_nG[:mynbands], kpt.q)

        for p1, (M1, M2) in enumerate(pairs_p):
            rho_MM[:] = 0.0
            rho_MM[M1,M2] = 1.0
            rho_MM = (rho_MM + rho_MM.T) / 2
            rho1_g, D1_ap = charge_density(self.wfs, self.density, rho_MM)

            if self.screening_omega == 0.0:
                VHt1_g = self.density.finegd.zeros()
                self.hamiltonian.poisson.solve(VHt1_g, rho1_g, charge=None)
            else:
                rho_g = finegd.zeros()
                #rho_g[:,:,:] = rho1_g
                rho_g[:-1,:-1,:-1] = rho1_g
                VHt1_G = pd12.fft(rho_g) * v_G

            for p2, (M3, M4) in enumerate(pairs_p):
                if p1 > p2:
                    continue
                print(p1, p2)
                rho_MM[:] = 0.0
                rho_MM[M3,M4] = 1.0
                rho_MM = (rho_MM + rho_MM.T) / 2
                rho2_g, D2_ap = charge_density(self.wfs, self.density, rho_MM)

                if self.screening_omega == 0.0:
                    K = self.density.finegd.integrate(VHt1_g * rho2_g)
                else:
                    rho_g = finegd.zeros()
                    rho_g[:-1,:-1,:-1] = rho2_g
                    #rho_g[:,:,:] = rho2_g
                    K = pd12.integrate(VHt1_G, pd12.fft(rho_g))

                K_MMMM[M1,M2,M3,M4] = K
                K_MMMM[M2,M1,M3,M4] = K
                K_MMMM[M2,M1,M4,M3] = K
                K_MMMM[M1,M2,M4,M3] = K
                K_MMMM[M3,M4,M1,M2] = K
                K_MMMM[M3,M4,M2,M1] = K
                K_MMMM[M4,M3,M2,M1] = K
                K_MMMM[M4,M3,M1,M2] = K
        self.K_kkMMMM[k12_c] = K_MMMM

    def nlxc(self, 
             H_MM:np.ndarray,
             dH_asp:Dict[int,np.ndarray],
             wfs,
             kpt) -> Tuple[float, float, float]:
        evc = 0.0
        evv = 0.0
        ekin = 0.0
        ekin = -2*evv -evc
        return evv, evc, ekin 

    def get_description(self):
        return 'Debug evaluation of full 4 center matrix elements'
