from __future__ import annotations

from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.smearing import OccupationNumberCalculator
from gpaw.new.pw.fulldiagclass import FullDiagonalizerComplex, FullDiagonalizerFloat


def diagonalize(potential: Potential,
                ibzwfs: IBZWaveFunctions,
                occ_calc: OccupationNumberCalculator,
                nbands: int | None,
                xc) -> IBZWaveFunctions:

    if ibzwfs.dtype == complex:
        diagonalizer = FullDiagonalizerComplex()
    elif ibzwfs.dtype == float:
        diagonalizer = FullDiagonalizerFloat()
    else:
        raise TypeError("one can use only float or complex dtype")

    return diagonalizer.diagonalize(potential, ibzwfs, occ_calc, nbands)
