import numpy as np
from scipy.linalg import eigh
from gpaw.new.eigensolver import Eigensolver

from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions


class LCAOEigensolver(Eigensolver):
    def __init__(self, basis):
        self.basis = basis

    def iterate(self, state, hamiltonian) -> float:
        matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            state)

        for wfs in state.ibzwfs:
            self.iterate1(wfs, matrix_calculator)
        return 0.0

    def iterate1(self,
                 wfs: LCAOWaveFunctions,
                 matrix_calculator: HamiltonianMatrixCalculator):
        H_MM = matrix_calculator.calculate_hamiltonian_matrix(wfs)

        eig_M, C_MM = eigh(H_MM, wfs.S_MM, overwrite_a=True, driver='gvd')
        wfs._eig_n = eig_M[:wfs.nbands]
        wfs.C_nM.data[:] = C_MM.T[:wfs.nbands]

        if wfs.dtype == complex:
            wfs.C_nM.complex_conjugate()

        # Make sure wfs.C_nM and (lacy) wfs.P_ain are in sync:
        wfs._P_ain = None
