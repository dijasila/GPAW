from typing import Tuple, Dict, Any, Sequence

import numpy as np

from gpaw import GPAW

Array2D = Any
Array4D = Any


class WannierOverlaps:
    def __init__(self,
                 cell: Sequence[Sequence[float]],
                 monkhorst_pack_size: Sequence[int],
                 directions: Dict[Tuple[int, int, int], int],
                 overlaps: Array4D):

        self.monkhorst_pack_size = tuple(monkhorst_pack_size)
        self.cell = np.array(cell)
        self.directions = directions
        nkpts, ndirs, self.nbands, nbands = overlaps.shape
        assert nbands == self.nbands
        assert nkpts == np.prod(monkhorst_pack_size)
        assert ndirs == len(directions)

        self._overlaps = overlaps

    def overlap(self,
                bz_index: int,
                direction: Tuple[int, int, int]) -> Array2D:
        return self._overlaps[bz_index, self.directions[direction]]

    def write(self, filename):
        ...


def calculate_overlaps(calc: GPAW,
                       n1: int = 0,
                       n2: int = 0,
                       soc: bool = False,
                       spin: int = 0) -> WannierOverlaps:
    if n2 <= 0:
        n2 += calc.get_number_of_bands()

    # world = calc.world
    wfs = calc.wfs
    kd = wfs.kd
    directions = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}
    Z_kdnn = np.empty((kd.nbzkpts, len(directions), n2 - n1, n2 - n1), complex)
    gd = wfs.gd
    u_knR = gd.empty((kd.nbzkpts, n2 - n1), complex)
    u_nR = gd.empty((n2 - n1), complex, global_array=True)
    for ibz_index in range(kd.nibzkpts):
        for n in range(n1, n2):
            u_nR[n - n1] = wfs.get_wave_function_array(n=n,
                                                       k=ibz_index,
                                                       s=spin,
                                                       periodic=True)
        u_knR[ibz_index] = u_nR

    size = kd.N_c

    for bz_index1 in range(kd.nbzkpts):
        u1_nR = u_knR[bz_index1]
        i1_c = np.unravel_index(bz_index1, size)
        for direction, d in directions.items():
            i2_c = np.array(i1_c) + direction
            bz_index2 = np.ravel_multi_index(i2_c, size, 'wrap')
            u2_nR = u_knR[bz_index2]
            phase_c = (i2_c % size - i2_c) // size
            if phase_c.any():
                u2_nR = u2_nR * gd.plane_wave(phase_c)
            Z_kdnn[bz_index1, d] = gd.integrate(u1_nR, u2_nR)

    overlaps = WannierOverlaps(calc.atoms.cell,
                               kd.N_c,
                               directions,
                               Z_kdnn)
    return overlaps
