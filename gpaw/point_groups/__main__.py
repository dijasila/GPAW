import sys

import numpy as np

from gpaw import GPAW
from gpaw.point_groups import SymmetryChecker


def main(filename, group_name, symbols, radius, n1, n2):
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
    checker.check_calculation(calc, n1, n2)


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise SystemExit(
            'Use:\n\n'
            '    python3 -m gpaw.point_groups '
            'gpw-file point-group-name center radius n1 n2')
    filename, group_name, symbols, radius, n1, n2 = sys.argv[1:]
    main(filename,
         group_name,
         set(symbols.split(',')),
         float(radius),
         int(n1), int(n2))
