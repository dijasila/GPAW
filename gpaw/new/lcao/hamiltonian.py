from __future__ import annotations

import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.new.calculation import DFTState
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.hamiltonian import Hamiltonian


class HamiltonianMatrixCalculator:

    def __init__(self,
                 V_sxMM: list[np.ndarray],
                 dH_saii: list[dict[int, np.ndarray]],
                 basis: BasisFunctions):
        self.V_sxMM = V_sxMM
        self.dH_saii = dH_saii
        self.basis = basis

    def calculate_potential_matrix(self,
                                   wfs: LCAOWaveFunctions) -> np.ndarray:
        V_xMM = self.V_sxMM[wfs.spin]
        V_MM = V_xMM[0]
        if wfs.dtype == complex:
            V_MM = V_MM.astype(complex)
            phase_x = np.exp(2j * np.pi *
                             self.basis.sdisp_xc[1:] @ wfs.kpt_c)
            V_MM += np.einsum('x, xMN -> MN',
                              2 * phase_x, V_xMM[1:],
                              optimize=True)
        return V_MM

    def calculate_hamiltonian_matrix(self,
                                     wfs: LCAOWaveFunctions) -> np.ndarray:
        H_MM = self.calculate_potential_matrix(wfs)
        for a, dH_ii in self.dH_saii[wfs.spin].items():
            P_Mi = wfs.P_aMi[a]
            H_MM += P_Mi @ dH_ii @ P_Mi.T.conj()

        if wfs.dtype == complex:
            H_MM *= 0.5
            H_MM += H_MM.conj().T

        H_MM += wfs.T_MM

        return H_MM


class LCAOHamiltonian(Hamiltonian):
    def __init__(self,
                 basis: BasisFunctions):
        self.basis = basis

    def create_hamiltonian_matrix_calculator(self,
                                             state: DFTState
                                             ) -> HamiltonianMatrixCalculator:
        V_sxMM = [self.basis.calculate_potential_matrices(vt_R.data)
                  for vt_R in state.potential.vt_sR]
        print(V_sxMM[0])

        dH_saii = [{a: dH_sii[s]
                    for a, dH_sii in state.potential.dH_asii.items()}
                   for s in range(len(V_sxMM))]
        print(dH_saii[0])

        return HamiltonianMatrixCalculator(V_sxMM, dH_saii, self.basis)
