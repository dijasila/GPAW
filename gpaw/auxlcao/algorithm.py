import numpy as np
from typing import Tuple, Dict
from gpaw.utilities import unpack2, packed_index, pack2

def rotate_density_matrix(rho_MM, symmetry, pointgroup_symmetry, time_reversal_symmetry):
    assert pointgroup_symmetry == 0
    # TODO
    return rho_MM


class RIAlgorithm:
    def __init__(self, name, exx_fraction, screening_omega):
        self.name = name
        self.exx_fraction = exx_fraction
        self.screening_omega = screening_omega

    def initialize(self, density, hamiltonian, wfs):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.timer = hamiltonian.timer
        for kpt in self.wfs.kpt_u:
            kpt.exx_V_MM = 0.0
        self.dH_asp = None

    def nlxc(self,
             H_MM:np.ndarray,
             dH_asp:Dict[int,np.ndarray],
             wfs,
             kpt, yy) -> Tuple[float, float, float]:

        #print(H_MM)
        H_MM += kpt.exx_V_MM * yy
        if self.dH_asp is None:
            print('Skipping dH_asp for first time')
        else:
            for a in self.dH_asp:
                #print('dH before ', dH_asp[a])
                print('Not adding dH_asp')
                #dH_asp[a] += self.dH_asp[a]
                #print('Added correction', self.dH_asp[a])
                #print('dH ', dH_asp[a])
        #print('Adding nlxc', kpt.exx_V_MM, H_MM)

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate_exchange_per_kpt_pair(self, kpt1, k_c, rho1_MM, kpt2, k2_c, rho2_MM):
        raise NotImplementedError

    def calculate_non_local(self):
         
        with self.timer('Calculate rho_MM'):
            for k, kpt in enumerate(self.wfs.kpt_u):
                kpt.exx_rho_MM = self.wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)

        kpt_u = self.wfs.kpt_u
        kd = self.wfs.kd
        evv = 0.0

        with self.timer('Hybrid'):
            for u, kpt in enumerate(kpt_u):
                kpt.exx_V_MM = 0.0
                for kbz, (kibz, pointgroup_symmetry, time_reversal_symmetry) in enumerate(zip(kd.bz2ibz_k, kd.sym_k, kd.time_reversal_k)):
                    kpt2 = kpt_u[kibz]
                    assert kpt2.q == kibz
                    kpt2_rho_MM = rotate_density_matrix(kpt2.exx_rho_MM, kd.symmetry, pointgroup_symmetry, time_reversal_symmetry)
                    E, V_MM = self.calculate_exchange_per_kpt_pair(kpt, kd.ibzk_qc[kpt.q], kpt.exx_rho_MM, kpt2, kd.bzk_kc[kbz], kpt2_rho_MM)
                    kpt.exx_V_MM += V_MM
                    evv += E

        evc = 0.0

        dH_asp = self.density.setups.empty_atomic_matrix(1, self.density.D_asp.partition, self.density.D_asp.dtype)
        with self.timer('RI Local atomic corrections'):
            for a in self.density.D_asp.keys():
                D_ii = unpack2(self.density.D_asp[a][0]) / 2 # Check 1 or 2
                # Copy-pasted from hybrids/pw.py
                ni = len(D_ii)
                V_ii = np.empty((ni, ni))
                for i1 in range(ni):
                    for i2 in range(ni):
                        V = 0.0
                        for i3 in range(ni):
                            p13 = packed_index(i1, i3, ni)
                            for i4 in range(ni):
                                p24 = packed_index(i2, i4, ni)
                                V += self.density.setups[a].M_pp[p13, p24] * D_ii[i3, i4]
                        V_ii[i1, i2] = +V
                V_p = pack2(V_ii)
                dH_asp[a][0][:] = (-V_p - self.density.setups[a].X_p) * self.exx_fraction

                evv -= self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2
                #print('EVV Correction', self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2)
                evc -= self.exx_fraction * np.dot(self.density.D_asp[a][0], self.density.setups[a].X_p)

   
        ekin = -2*evv -evc
        #print(evv,evc,ekin)
        self.dH_asp = dH_asp
        return evv, evc, ekin

