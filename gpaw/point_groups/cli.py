import argparse
from typing import List, Any

import numpy as np

from gpaw.point_groups import SymmetryChecker, point_group_names

Array1D = Any
Array3D = Any


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('pg', metavar='point-group', choices=point_group_names)
    add('file', metavar='input-file')
    add('-c', '--center')
    add('-r', '--radius', default=2.5)
    add('-b', '--bands', default=':')
    args = parser.parse_args(argv)

    if args.file.endswith('.gpw'):
        from gpaw import GPAW
        calc = GPAW(args.file)
        atoms = calc.atoms
        n1, n2 = (int(x) if x else 0 for x in args.bands.split(':'))
    elif args.file.endswith('.cube'):
        from ase.io.cube import read_cube
        dct = read_cube(args.file)
        calc = CubeCalc(dct['data'])
        atoms = dct['atoms']
        n1 = 0
        n2 = 1
    else:
        from ase.io import read
        atoms = read(args.file)
        calc = None

    if args.center:
        symbols = set(args.center.split(','))
        center = np.zeros(3)
        n = 0
        for symbol, position in zip(atoms.symbols, atoms.positions):
            if symbol in symbols:
                center += position
                n += 1
        center /= n
    else:
        center = atoms.cell.sum(0) / 2
    print('Center:', center)

    radius = float(args.radius)

    checker = SymmetryChecker(args.pg, center, radius)

    ok = checker.check_atoms(atoms)
    print(f'{args.pg}-symmetry:', 'Yes' if ok else 'No')

    if calc:
        checker.check_calculation(calc, n1, n2)


class CubeCalc:
    def __init__(self, function: Array3D):
        self.function = function

    def get_pseudo_wave_function(self,
                                 band: int,
                                 spin: int,
                                 pad: bool) -> Array3D:
        return self.function

    def get_eigenvalues(self, spin: int) -> Array1D:
        return np.zeros(1)
