from math import inf, nan
from typing import List, Tuple

import numpy as np

from gpaw.occupations import OccupationNumberCalculator
from gpaw.hints import Array1D


class DeltaSCF(OccupationNumberCalculator):
    extrapolate_factor = 0.0

    def __init__(self, neup, nedown, excitations, atoms):
        self.neup = neup
        self.nedown = nedown
        self.excitations = excitations
        self.atoms = atoms

        OccupationNumberCalculator.__init__(self)

        self.signatures: List[Tuple[int, float, Array1D]] = []
        self.bands = []

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: List[List[float]],
                  weights: List[float],
                  fermi_levels_guess: List[float] = None
                  ) -> Tuple[List[np.ndarray],
                             List[float],
                             float]:
        bd = self.bd

        P_snI = [self.atoms.calc.wfs.kpt_u[s].projections.array
                 for s in [0, 1]]

        if not self.signatures:
            for spin, band, occ in self.excitations:
                rank, myn = bd.who_has(band)
                if bd.comm.rank == rank:
                    P_I = P_snI[spin][myn].copy()
                else:
                    P_I = np.zeros(P_snI[0].shape[1])
                bd.comm.broadcast(P_I, rank)
                self.signatures.append((spin, occ, P_I))

        self.bands = [
            find_band(self.bd, self.domain_comm, P_snI[spin], P_I)
            for spin, occ, P_I in self.signatures]

        return OccupationNumberCalculator.calculate(self,
                                                    nelectrons,
                                                    eigenvalues,
                                                    weights,
                                                    fermi_levels_guess)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess=nan):
        f_sn = self.bd.zeros(2)
        f_sn[0, :self.neup] = 1.0
        f_sn[1, :self.nedown] = 1.0

        for band, signature in zip(self.bands, self.signatures):
            spin, occ, _ = signature
            f_sn[spin, band] += occ

        for F_n, f_n in zip(f_sn, f_qn):
            self.bd.distribute(F_n, f_n)

        return inf, 0.0


def find_band(bd, domain_comm, P_nI, P_I) -> int:
    P_n = P_nI.dot(P_I)
    domain_comm.summ(P_n)
    P_n *= P_n
    P_n = bd.collect(P_n, broadcast=True)
    return P_n.argmax()
