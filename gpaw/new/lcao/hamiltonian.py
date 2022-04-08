from __future__ import annotations

import numpy as np

from gpaw.external import ExternalPotential
from gpaw.lfc import BasisFunctions
from gpaw.new.calculation import DFTState
from gpaw.new.fd.pot_calc import UniformGridPotentialCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.core.matrix import Matrix


class HamiltonianMatrixCalculator:

    def __init__(self,
                 V_sxMM: list[np.ndarray],
                 dH_saii: list[dict[int, np.ndarray]],
                 basis: BasisFunctions):
        self.V_sxMM = V_sxMM
        self.dH_saii = dH_saii
        self.basis = basis

    def calculate_potential_matrix(self,
                                   wfs: LCAOWaveFunctions) -> Matrix:
        V_xMM = self.V_sxMM[wfs.spin]
        V_MM = V_xMM[0]
        if wfs.dtype == complex:
            V_MM = V_MM.astype(complex)
            phase_x = np.exp(2j * np.pi *
                             self.basis.sdisp_xc[1:] @ wfs.kpt_c)
            V_MM += np.einsum('x, xMN -> MN',
                              2 * phase_x, V_xMM[1:],
                              optimize=True)

        _, M = V_MM.shape
        matrix_V_MM = Matrix(M, M, data=V_MM, dist=(wfs.band_comm,))

        M = matrix_V_MM.dist.myslice()
        for a, dH_ii in self.dH_saii[wfs.spin].items():
            P_Mi = wfs.P_aMi[a]
            matrix_V_MM.data += P_Mi[M] @ dH_ii @ P_Mi.T.conj()
        wfs.domain_comm.sum(matrix_V_MM.data)

        return matrix_V_MM

    def calculate_hamiltonian_matrix(self,
                                     wfs: LCAOWaveFunctions) -> Matrix:
        H_MM = self.calculate_potential_matrix(wfs)
        if wfs.dtype == complex:
            H_MM.add_hermitian_conjugate(scale=0.5)
        else:
            H_MM.tril2full()

        H_MM.data += wfs.T_MM

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

        dH_saii = [{a: dH_sii[s]
                    for a, dH_sii in state.potential.dH_asii.items()}
                   for s in range(len(V_sxMM))]

        return HamiltonianMatrixCalculator(V_sxMM, dH_saii, self.basis)

    def create_kick_matrix_calculator(self,
                                      state: DFTState,
                                      ext: ExternalPotential,
                                      pot_calc: UniformGridPotentialCalculator
                                      ) -> HamiltonianMatrixCalculator:
        from gpaw.utilities import unpack
        vext_r = pot_calc.vbar_r.new()
        finegd = vext_r.desc._gd

        vext_r.data = ext.get_potential(finegd)
        vext_R = pot_calc.restrict(vext_r)

        nspins = state.ibzwfs.nspins

        V_MM = self.basis.calculate_potential_matrices(vext_R.data)
        V_sxMM = [V_MM for s in range(nspins)]

        W_aL = pot_calc.ghat_aLr.integrate(vext_r)

        assert state.ibzwfs.ibz.bz.gamma_only
        setups_a = state.ibzwfs.wfs_qs[0][0].setups

        dH_saii = [{a: unpack(setups_a[a].Delta_pL @ W_L)
                    for (a, W_L) in W_aL.items()}
                   for s in range(nspins)]

        return HamiltonianMatrixCalculator(V_sxMM, dH_saii, self.basis)
