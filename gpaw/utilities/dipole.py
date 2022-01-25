"""Calculate dipole matrix elements."""

from math import pi
from typing import List, Iterable

from ase.units import Bohr
import numpy as np

from gpaw.calculator import GPAW
from gpaw.grid_descriptor import GridDescriptor
from gpaw.setup import Setup
from gpaw.mpi import serial_comm
from gpaw.typing import Array2D, Array3D, Array4D, Vector


def dipole_matrix_elements(gd: GridDescriptor,
                           psit_nR: List[Array3D],
                           P_nI: Array2D,
                           position_av: Array2D,
                           setups: List[Setup],
                           center: Vector) -> Array3D:
    """Calculate dipole matrix-elements.

    gd:
        Grid-descriptor.
    psit_nG:
        Wave functions in atomic units.
    P_nI:
        PAW projections.
    setups:
        PAW setups.

    Returns matrix elements in atomic units.
    """
    assert gd.comm.size == 1
    dipole_nnv = np.empty((len(psit_nR), len(psit_nR), 3))

    for na, psita_R in enumerate(psit_nR):
        for nb, psitb_R in enumerate(psit_nR[:na + 1]):
            d_v = -gd.dipole_moment(psita_R * psitb_R, center)
            dipole_nnv[na, nb] = d_v
            dipole_nnv[nb, na] = d_v

    scenter_c = np.linalg.solve(gd.cell_cv.T, center)
    spos_ac = np.linalg.solve(gd.cell_cv.T, position_av.T).T % 1.0
    spos_ac -= scenter_c - 0.5
    spos_ac %= 1.0
    spos_ac += scenter_c - 0.5
    position_av = spos_ac.dot(gd.cell_cv)

    R_aiiv = []
    for setup, position_v in zip(setups, position_av):
        Delta_iiL = setup.Delta_iiL
        R_iiv = Delta_iiL[:, :, [3, 1, 2]] * (4 * pi / 3)**0.5
        R_iiv += position_v * setup.Delta_iiL[:, :, :1] * (4 * pi)**0.5
        R_aiiv.append(R_iiv)

    I1 = 0
    for R_iiv in R_aiiv:
        I2 = I1 + len(R_iiv)
        P_ni = P_nI[:, I1:I2]
        dipole_nnv += np.einsum('mi, ijv, nj -> mnv', P_ni, R_iiv, P_ni)
        I1 = I2

    return dipole_nnv


def dipole_matrix_elements_from_calc(calc: GPAW,
                                     n1: int,
                                     n2: int,
                                     center: Iterable[float] = None
                                     ) -> List[Array4D]:
    """Calculate dipole matrix-elements (units: Å)."""
    wfs = calc.wfs
    assert wfs.kd.gamma

    gd = wfs.gd.new_descriptor(comm=serial_comm)
    position_av = calc.atoms.positions / Bohr

    if center is None:
        center_v = gd.cell_cv.sum(axis=0) / 2
    else:
        center_v = np.asarray(center) / Bohr

    d_snnv = []
    for spin in range(calc.get_number_of_spins()):
        psit_nR = [calc.get_pseudo_wave_function(band=n, spin=spin,
                                                 pad=False) * Bohr**1.5
                   for n in range(n1, n2)]
        projections = wfs.kpt_qs[0][spin].projections.view(n1, n2)
        P_nI = projections.collect()

        if wfs.world.rank == 0:
            d_nnv = dipole_matrix_elements(gd,
                                           psit_nR,
                                           P_nI,
                                           position_av,
                                           wfs.setups,
                                           center_v) * Bohr
        else:
            d_nnv = np.empty((n2 - n1, n2 - n1, 3))
        wfs.world.broadcast(d_nnv, 0)
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

    if calc.wfs.world.rank > 0:
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
