"""Edmiston-ruedenberg localization."""
from math import pi
from typing import Any

import numpy as np

import _gpaw
from .overlaps import WannierOverlaps
from .functions import WannierFunctions

Array2D = Any


class LocalizationNotConvergedError(Exception):
    """Error raised if maxiter is exceeded."""


def localize(overlaps: WannierOverlaps,
             maxiter: int = 100,
             tolerance: float = 1e-5,
             verbose: bool = not False) -> WannierFunctions:
    """Simple localization.

    Orthorhombic cell, no k-points, only occupied states.
    """
    if not overlaps.atoms.cell.orthorhombic:
        raise NotImplementedError('An orthogonal cell is required')
    assert overlaps.monkhorst_pack_size == (1, 1, 1)

    Z_nnc = np.empty((overlaps.nbands, overlaps.nbands, 3), complex)
    for c, direction in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        Z_nnc[:, :, c] = overlaps.overlap(bz_index=0, direction=direction)

    U_nn = np.identity(overlaps.nbands)

    if verbose:
        print('iter      value     change')
        print('---- ---------- ----------')

    old = 0.0
    for iter in range(maxiter):
        value = _gpaw.localize(Z_nnc, U_nn)
        if verbose:
            print(f'{iter:4} {value:10.3f} {value - old:10.6f}')
        if value - old < tolerance:
            break
        old = value
    else:
        raise LocalizationNotConvergedError(
            f'Did not converge in {maxiter} iterations')

    # Find centers:
    scaled_nc = -np.angle(Z_nnc.diagonal()).T / (2 * pi)
    centers_nv = (scaled_nc % 1.0).dot(overlaps.atoms.cell)

    return WannierFunctions(overlaps.atoms, U_nn, centers_nv, value)


if __name__ == '__main__':
    import sys
    from gpaw import GPAW
    from .overlaps import calculate_overlaps

    calc = GPAW(sys.argv[1])
    overlaps = calculate_overlaps(calc)
    wan = overlaps.localize(verbose=True)
    print(wan.centers)
    wan.centers_as_atoms().edit()
