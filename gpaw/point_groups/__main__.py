import sys

import numpy as np

from gpaw import GPAW
from gpaw.point_groups import SymmetryChecker


def main(filename, group_name, symbols, radius):
    calc = GPAW(filename)
    atoms = calc.atoms
    center = np.zeros(3)
    n = 0
    for symbol, position in zip(atoms.symbols, atoms.positions):
        if symbol in symbols:
            center += position
            n += 1
    center /= n
    checker = SymmetryChecker(group_name, center, radius)
    checker.check_calculation(calc, 0, calc.get_number_of_bands())


if __name__ == '__main__':
    filename, group_name, symbols, radius = sys.argv[1:]
    main(filename, group_name, set(symbols.split(',')), float(radius))
