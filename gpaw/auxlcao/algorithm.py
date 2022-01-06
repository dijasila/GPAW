import numpy as np
from typing import Tuple, Dict


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

    def nlxc(self,
             H_MM:np.ndarray,
             dH_asp:Dict[int,np.ndarray],
             wfs,
             kpt) -> Tuple[float, float, float]:
        H_MM += kpt.exx_V_MM

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate_exchange_per_kpt_pair(self, k_c, k2_c, rho2_MM):
        raise NotImplementedError

        print('Calculating ', k_c, k2_c)
        return 0.0

    def calculate_non_local(self):
         
        with self.timer('Calculate rho_MM'):
            for k, kpt in enumerate(self.wfs.kpt_u):
                kpt.exx_rho_MM = self.wfs.ksl.calculate_density_matrix(kpt.f_n, kpt.C_nM)

        kpt_u = self.wfs.kpt_u
        kd = self.wfs.kd

        with self.timer('Hybrid'):
            for u, kpt in enumerate(kpt_u):
                kpt.exx_V_MM = 0.0
                for kbz, (kibz, pointgroup_symmetry, time_reversal_symmetry) in enumerate(zip(kd.bz2ibz_k, kd.sym_k, kd.time_reversal_k)):
                    kpt2 = kpt_u[kibz]
                    assert kpt2.q == kibz
                    kpt2_rho_MM = rotate_density_matrix(kpt2.exx_rho_MM, kd.symmetry, pointgroup_symmetry, time_reversal_symmetry)
                    kpt.exx_V_MM += self.calculate_exchange_per_kpt_pair(kd.ibzk_qc[kpt.q], kd.bzk_kc[kbz], kpt2_rho_MM)

        evc = 0.0
        evv = 0.0
        ekin = 0.0
        ekin = -2*evv -evc
        return evv, evc, ekin

