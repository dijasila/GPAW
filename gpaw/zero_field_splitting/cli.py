from typing import List, Tuple
import numpy as np
from ase.units import _c, _e, _hplanck
from gpaw.typing import Array1D, Array2D
from gpaw.zero_field_splitting import zfs


def convert_tensor(D_vv: Array2D,
                   unit: str = 'eV') -> Tuple[float, float, Array1D, Array2D]:
    """Convert 3x3 tensor to D, E and easy axis.

    Input tensor must be in eV and the result can be returned in
    eV, μeV, MHz or 1/cm acording to the value uf *unit*
    (must be one of "eV", "ueV", "MHz", "1/cm").

    >>> D_vv = np.diag([1, 2, 3])
    >>> D, E, axis, _ = convert_tensor(D_vv)
    >>> D
    4.5
    >>> E
    0.5
    >>> axis
    array([0., 0., 1.])
    """
    if unit == 'ueV':
        scale = 1e6
    elif unit == 'MHz':
        scale = _e / _hplanck * 1e-6
    elif unit == '1/cm':
        scale = _e / _hplanck / _c / 100
    elif unit == 'eV':
        scale = 1.0
    else:
        raise ValueError(f'Unknown unit: {unit}')

    (e1, e2, e3), U = np.linalg.eigh(D_vv * scale)

    if abs(e1) > abs(e3):
        D = 1.5 * e1
        E = 0.5 * (e2 - e3)
        axis = U[:, 0]
    else:
        D = 1.5 * e3
        E = 0.5 * (e2 - e1)
        axis = U[:, 2]

    return D, E, axis, D_vv * scale


def main(argv: List[str] = None) -> Array2D:
    """CLI interface."""
    import argparse

    from gpaw import GPAW
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.zero_field_splitting',
        description='...')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-u', '--unit', default='ueV', choices=['eV', 'ueV', 'MHz', '1/cm'],
        help='Unit.  Must be "eV", "ueV" (micro-eV, default), '
        '"MHz" or "1/cm".')
    add('-m', '--method', type=int, default=1)
    add('-g', '--grid-spacing', type=float, default=-1.0)

    if hasattr(parser, 'parse_intermixed_args'):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    calc = GPAW(args.file)

    D_vv = zfs(calc, args.method, args.grid_spacing != -1.0, args.grid_spacing)
    D, E, axis, D_vv = convert_tensor(D_vv, args.unit)

    unit = args.unit
    if unit == 'ueV':
        unit = 'μeV'

    print('D_ij = (' +
          ',\n        '.join('(' + ', '.join(f'{d:10.3f}' for d in D_v) + ')'
                             for D_v in D_vv) +
          ') ', unit)
    print('i, j = x, y, z')
    print()
    print(f'D = {D:.3f} {unit}')
    print(f'E = {E:.3f} {unit}')
    x, y, z = axis
    print(f'axis = ({x:.3f}, {y:.3f}, {z:.3f})')

    return D_vv


if __name__ == '__main__':
    main()
