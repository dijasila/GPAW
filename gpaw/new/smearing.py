from __future__ import annotations
from gpaw.occupations import create_occ_calc, ParallelLayout
from gpaw.band_descriptor import BandDescriptor
from gpaw.typing import ArrayLike2D, Array2D


class OccupationNumberCalculator:
    def __init__(self, dct, pbc, ibz, nbands, comms, magmoms, rcell):
        if dct is None:
            if pbc.any():
                dct = {'name': 'fermi-dirac',
                       'width': 0.1}  # eV
            else:
                dct = {'width': 0.0}

        if magmoms is None:
            dct.pop('fixmagmom', None)
            magmom = 0.0
        else:
            magmom = magmoms.sum(0)

        kwargs = dct.copy()
        name = kwargs.pop('name', '')
        if name == 'mom':
            from gpaw.mom import OccupationsMOM
            return OccupationsMOM(..., **kwargs)

        self.band_comm = comms['b']
        bd = BandDescriptor(nbands)
        self.occ = create_occ_calc(
            dct,
            parallel_layout=ParallelLayout(bd,
                                           comms['k'],
                                           comms['d']),
            fixed_magmom_value=magmom,
            rcell=rcell,
            monkhorst_pack_size=ibz.bz.size_c,
            bz2ibzmap=ibz.bz2ibz_K)
        self.extrapolate_factor = self.occ.extrapolate_factor

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: ArrayLike2D,
                  weights: list[float],
                  fermi_levels_guess: list[float] = None
                  ) -> tuple[Array2D, list[float], float]:
        if self.band_comm.rank == 0:
            occs, fls, e = self.occ.calculate(nelectrons, eigenvalues, weights,
                                              fermi_levels_guess)
        else:
            1 / 0
        return occs, fls, e
