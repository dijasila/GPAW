"""Calculate dipole matrix elements."""

from typing import List, Any

from ase.units import Bohr
import numpy as np

from gpaw.grid_descriptor import GridDescriptor
from gpaw.projections import Projections
from gpaw import GPAW

Array3D = Any
Array4D = Any


def dipole_matrix_elements(gd: GridDescriptor,
                           psit_nR: Array4D,
                           projections: Projections) -> Array3D:
    dipole_nnv = np.empty((len(psit_nR), len(psit_nR), 3))
    for na, psita_R in enumerate(psit_nR):
        for nb, psitb_R in enumerate(psit_nR[:na + 1]):
            d_v = gd.calculate_dipole_moment(psita_R * psitb_R)

            dipole_nnv[na, nb] = d_v
            dipole_nnv[nb, na] = d_v
    return dipole_nnv


def main(argv: List[str] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.utilities.dipole',
        description='Calculate dipole matrix elements.')

    add = parser.add_argument

    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-n', '--band-range', nargs=2, type=int)

    if hasattr(parser, 'parse_intermixed_args'):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    calc = GPAW(args.file)

    n1, n2 = args.band_range

    wfs = calc.wfs
    psit_nR = np.array([calc.get_pseudo_wave_function(band=n)
                        for n in range(n1, n2)])
    psit_nR = wfs.gd.distribute(psit_nR) * Bohr**1.5
    projections = wfs.kpt_u[0].projections.view(n1, n2)
    d_nnv = dipole_matrix_elements(wfs.gd,
                                   psit_nR,
                                   projections) * Bohr
    print(d_nnv)


if __name__ == '__main__':
    main()
