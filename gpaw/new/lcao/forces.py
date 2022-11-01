from __future__ import annotations
from typing import Tuple

import numpy as np
from gpaw.lfc import BasisFunctions
from gpaw.new import zip
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.potential import Potential
from gpaw.typing import Array2D, Array3D
from gpaw.core.uniform_grid import UniformGrid

Derivatives = Tuple[Array3D,
                    Array3D,
                    dict[int, Array3D]]


class TCIDerivatives:
    def __init__(self, manytci, atomdist, nao: int):
        self.manytci = manytci
        self.atomdist = atomdist
        self.nao = nao

        self._derivatives_q: dict[int, Derivatives] = {}

    def calculate_derivatives(self,
                              q: int) -> Derivatives:
        if not self._derivatives_q:
            dThetadR_qvMM, dTdR_qvMM = self.manytci.O_qMM_T_qMM(
                self.atomdist.comm,
                0, self.nao,
                False, derivative=True)

            self.atomdist.comm.sum(dThetadR_qvMM)
            self.atomdist.comm.sum(dTdR_qvMM)

            dPdR_aqvMi = self.manytci.P_aqMi(
                self.atomdist.indices,
                derivative=True)

            dPdR_qavMi = [{a: dPdR_qvMi[q]
                           for a, dPdR_qvMi in dPdR_aqvMi.items()}
                          for q in range(len(dThetadR_qvMM))]

            self._derivatives_q = {
                q: (dThetadR_vMM, dTdR_vMM, dPdR_avMi)
                for q, (dThetadR_vMM, dTdR_vMM, dPdR_avMi)
                in enumerate(zip(dThetadR_qvMM, dTdR_qvMM, dPdR_qavMi))}

        return self._derivatives_q[q]


def add_force_contributions(wfs: LCAOWaveFunctions,
                            potential: Potential,
                            F_av: Array2D) -> None:
    (dThetadR_vMM,
     dTdR_vMM,
     dPdR_avMi) = wfs.tci_derivatives.calculate_derivatives(wfs.q)

    indices = []
    M1 = 0
    for a, P_Mi in wfs.P_aMi.items():
        M2 = M1 + len(P_Mi)
        indices.append((a, M1, M2))
        M1 = M2

    # Transpose?
    rhoT_MM = wfs.calculate_density_matrix().T
    erhoT_MM = wfs.calculate_density_matrix(eigs=True).T

    add_kinetic_term(rhoT_MM, dTdR_vMM, F_av, indices)
    add_pot_term(potential.vt_sR[wfs.spin], wfs.basis, wfs.q, rhoT_MM, F_av)
    add_den_mat_term(erhoT_MM, dThetadR_vMM, F_av, indices)
    add_den_mat_paw_term()
    add_atomic_density_term()


def add_kinetic_term(rho_MM, dTdR_vMM, F_av, indices):
    """Calculate Kinetic energy term in LCAO

    :::

                      dT
     _a        --- --   μν
     F += 2 Re >   >  ---- ρ
               --- --  _    νμ
               μ=a ν  dR
                        μν
            """

    for a, M1, M2 in indices:
        F_av[a, :] += 2 * np.einsum('vmM, Mm -> v',
                                    dTdR_vMM[:, M1:M2],
                                    rho_MM[:, M1:M2])


def add_pot_term(vt_R: UniformGrid,
                 basis,
                 q: int,
                 rhoT_MM,
                 F_av) -> None:
    """Calculate potential term"""
    # Potential contribution
    #
    #           -----      /  d Phi  (r)
    #  a         \        |        mu    ~
    # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
    #            /        |     d R                nu          nu mu
    #           -----    /         a
    #        mu in a; nu
    #
    F_av += basis.calculate_force_contribution(vt_R.data,
                                               rhoT_MM,
                                               q)


def add_den_mat_term(erho_MM, dThetadR_vMM, F_av, indices):
    """Calculate density matrix term in LCAO"""
    # Density matrix contribution due to basis overlap
    #
    #            ----- d Theta
    #  a          \           mu nu
    # F  += -2 Re  )   ------------  E
    #             /        d R        nu mu
    #            -----        mu nu
    #         mu in a; nu
    #
    for a, M1, M2 in indices:
        F_av[a, :] -= 2 * np.einsum('vmM, Mm -> v',
                                    dThetadR_vMM[:, M1:M2],
                                    erho_MM[:, M1:M2])


def add_den_mat_paw_term(setups, ):
    """Calcualte PAW correction"""
    # TO DO: split this function into
    # _get_den_mat_paw_term (which calculate Frho_av) and
    # get_paw_correction (which calculate ZE_MM)
    # Density matrix contribution from PAW correction
    #
    #           -----                        -----
    #  a         \      a                     \     b
    # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
    #            /      mu nu  nu mu          /     mu nu  nu mu
    #           -----                        -----
    #           mu nu                    b; mu in a; nu
    #
    # with
    #                  b*
    #         -----  dP
    #   b      \       i mu    b   b
    #  Z     =  )   -------- dS   P
    #   mu nu  /     dR        ij  j nu
    #         -----    b mu
    #           ij
    #
    work_MM = np.zeros((self.mynao, self.nao), self.dtype)
    ZE_MM = None
    for b in self.my_atom_indices:
        setup = self.setups[b]
        dO_ii = np.asarray(setup.dO_ii, self.dtype)
        dOP_iM = np.zeros((setup.ni, self.nao), self.dtype)
        mmm(1.0, dO_ii, 'N', self.P_aqMi[b][kpt.q], 'C', 0.0, dOP_iM)
        for v in range(3):
            mmm(1.0,
                self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop],
                'N',
                dOP_iM, 'N',
                0.0, work_MM)
            ZE_MM = (work_MM * self.ET_uMM[u]).real
            for a, M1, M2 in self.slices():
                dE = 2 * ZE_MM[M1:M2].sum()
                Frho_av[a, v] -= dE  # the "b; mu in a; nu" term
                Frho_av[b, v] += dE  # the "mu nu" term


def add_atomic_density_term(self):
    # Atomic density contribution
    #            -----                         -----
    #  a          \     a                       \     b
    # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
    #             /     mu nu    nu mu          /     mu nu    nu mu
    #            -----                         -----
    #            mu nu                     b; mu in a; nu
    #
    #                  b*
    #         ----- d P
    #  b       \       i mu   b   b
    # A     =   )   ------- dH   P
    #  mu nu   /    d R       ij  j nu
    #         -----    b mu
    #           ij
    #
    for b in self.my_atom_indices:
        H_ii = np.asarray(unpack(self.dH_asp[b][kpt.s]), self.dtype)
        HP_iM = gemmdot(H_ii, np.ascontiguousarray(
                        self.P_aqMi[b][kpt.q].T.conj()))
        for v in range(3):
            dPdR_Mi = \
                self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop]
            ArhoT_MM = \
                (gemmdot(dPdR_Mi, HP_iM) * self.rhoT_uMM[u]).real
            for a, M1, M2 in self.slices():
                dE = 2 * ArhoT_MM[M1:M2].sum()
                Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                Fatom_av[b, v] -= dE  # the "mu nu" term
