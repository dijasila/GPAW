"""Calculate dipole matrix elements."""

from math import pi
from typing import List, Iterable

from ase.units import Bohr
import numpy as np

from gpaw import GPAW
from gpaw.grid_descriptor import GridDescriptor
from gpaw.setup import Setup
from gpaw.mpi import serial_comm
from gpaw.typing import Array2D, Array3D, Array4D, Vector


def dipole_matrix_elements_from_calc(calc: GPAW,
                                     n1: int,
                                     n2: int,
                                     center_v: Vector = None
                                     ) -> List[Array4D]:
    """Calculate dipole matrix-elements (units: Å)."""
    ibzwfs = calc.calculation.state.ibzwfs
    assert ibzwfs.ibz.bz.gamma_only
    assert ibzwfs.band_comm.size == 1

    wfs_s = ibzwfs.wfs_qs[0]
    d_snnv = []
    for wfs in wfs_s:
        wfs = wfs.collect(n1, n2)
        if wfs is not None:
            d_nnv = wfs.dipole_matrix_elements(center_v) * Bohr
        else:
            d_nnv = np.empty((n2 - n1, n2 - n1, 3))
        calc.params.parallel['world'].broadcast(d_nnv, 0)
        d_snnv.append(d_nnv)

    return d_snnv


def main(argv: List[str] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.utilities.dipole',
        description='Calculate dipole matrix elements.  Units: Å.')

    add = parser.add_argument

    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-n', '--band-range', nargs=2, type=int, default=[0, 0],
        metavar=('n1', 'n2'), help='Include bands: n1 <= n < n2.')
    add('-c', '--center', metavar='x,y,z',
        help='Center of charge distribution.  Default is middle of unit '
        'cell.')

    args = parser.parse_intermixed_args(argv)

    calc = GPAW(args.file)

    n1, n2 = args.band_range
    nbands = calc.get_number_of_bands()
    n2 = n2 or n2 + nbands

    if args.center:
        center = [float(x) for x in args.center.split(',')]
    else:
        center = calc.atoms.cell.sum(axis=0) / 2  # center of cell

    d_snnv = dipole_matrix_elements_from_calc(calc, n1, n2, center)

    if calc.params.parallel['world'].rank > 0:
        return

    print('Number of bands:', nbands)
    print('Number of valence electrons:', calc.get_number_of_electrons())
    print()

    for spin, d_nnv in enumerate(d_snnv):
        print(f'Spin={spin}')

        for direction, d_nn in zip('xyz', d_nnv.T):
            print(f' <{direction}>',
                  ''.join(f'{n:8}' for n in range(n1, n2)))
            for n in range(n1, n2):
                print(f'{n:4}', ''.join(f'{d:8.3f}' for d in d_nn[n - n1]))


if __name__ == '__main__':
    main()
