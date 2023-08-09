from __future__ import annotations
from gpaw.occupations import create_occ_calc, ParallelLayout
from gpaw.band_descriptor import BandDescriptor
from gpaw.typing import ArrayLike2D, Array2D


class OccupationNumberCalculator:
    def __init__(self,
                 dct,
                 pbc,
                 ibz,
                 nbands,
                 comms,
                 magmom_v,
                 ncomponents,
                 rcell):
        if dct is None:
            if pbc.any():
                dct = {'name': 'fermi-dirac',
                       'width': 0.1}  # eV
            else:
                dct = {'width': 0.0}

        if dct.get('fixmagmom'):
            if ncomponents == 1:
                dct = dct.copy()
                del dct['fixmagmom']
            assert ncomponents == 2

        kwargs = dct.copy()
        name = kwargs.pop('name', '')
        if name == 'mom':
            1 / 0
            # from gpaw.mom import OccupationsMOM
            # return OccupationsMOM(..., **kwargs)

        bd = BandDescriptor(nbands)  # dummy
        self.occ = create_occ_calc(
            dct,
            parallel_layout=ParallelLayout(bd,
                                           comms['k'],
                                           comms['K']),
            fixed_magmom_value=magmom_v[2],
            rcell=rcell,
            monkhorst_pack_size=getattr(ibz.bz, 'size_c', None),
            bz2ibzmap=ibz.bz2ibz_K)
        self.extrapolate_factor = self.occ.extrapolate_factor

    def __str__(self):
        return str(self.occ)

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: ArrayLike2D,
                  weights: list[float],
                  fermi_levels_guess: list[float] = None
                  ) -> tuple[Array2D, list[float], float]:
        occs, fls, e = self.occ.calculate(nelectrons, eigenvalues, weights,
                                          fermi_levels_guess)
        return occs, fls, e
