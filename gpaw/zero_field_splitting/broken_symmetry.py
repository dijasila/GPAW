from math import inf, nan
from typing import List, Tuple

import numpy as np
from ase import Atoms

from gpaw.occupations import OccupationNumberCalculator
from gpaw.typing import Array1D


class BS(OccupationNumberCalculator):
    extrapolate_factor = 0.0

    def __init__(self,
                 neup: int,
                 nedown: int,
                 excitations: List[Tuple[int, int, float]],
                 atoms: Atoms):
        """Excited occupation numbers."""
        self.neup = neup
        self.nedown = nedown
        self.excitations = excitations
        self.atoms = atoms

        OccupationNumberCalculator.__init__(self)

        self.signatures: List[Tuple[int, float, Array1D]] = []
        self.bands: List[int] = []
        self.niter = 0

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: List[List[float]],
                  weights: List[float],
                  fermi_levels_guess: List[float] = None
                  ) -> Tuple[np.ndarray,
                             List[float],
                             float]:
        wfs = self.atoms.calc.wfs

        P_snI = [wfs.kpt_u[s].projections.array
                 for s in [0, 1]]

        if self.niter == 0:
            self.bd = wfs.bd
            self.kpt_comm = wfs.kd.comm
            self.domain_comm = wfs.gd.comm

        if not self.signatures and self.niter > 20:
            for spin, band, occ in self.excitations:
                rank, myn = self.bd.who_has(band)
                if self.bd.comm.rank == rank:
                    P_I = P_snI[spin][myn].copy()
                else:
                    P_I = np.zeros(P_snI[0].shape[1])
                self.bd.comm.broadcast(P_I, rank)
                self.signatures.append((spin, occ, P_I))

        if self.signatures:
            self.bands = [
                find_band(self.bd, self.domain_comm, P_snI[spin], P_I)
                for spin, occ, P_I in self.signatures]

        self.atoms.calc.log('Delta-SCF bands:', self.bands)

        self.niter += 1

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
        f_sn = [self.bd.zeros(global_array=True),
                self.bd.zeros(global_array=True)]
        f_sn[0][:self.neup] = 1.0
        f_sn[1][:self.nedown] = 1.0

        if self.signatures:
            for band, signature in zip(self.bands, self.signatures):
                spin, occ, _ = signature
                f_sn[spin][band] += occ
        else:
            f_sn[0][self.neup] = 1.0
            f_sn[1][self.nedown] = 1.0

        for F_n, f_n in zip(f_sn, f_qn):
            self.bd.distribute(F_n, f_n)

        return inf, 0.0


def find_band(bd, domain_comm, P_nI, P_I) -> int:
    P_n = P_nI.dot(P_I)
    domain_comm.sum(P_n)
    P_n *= P_n
    P_n = bd.collect(P_n, broadcast=True)
    if domain_comm.rank == 0:
        print(np.argsort(P_n)[-10:])
        print(sorted(P_n)[-10:])
    return P_n.argmax()
