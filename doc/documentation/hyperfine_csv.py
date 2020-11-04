# creates: g-factors.csv
from math import pi
import ase.units as units
from gpaw.hyperfine import gyromagnetic_ratios

with open('g-factors.csv', 'w') as fd:
    print('Nucleus, g-factor', file=fd)
    for symbol, (n, ratio) in gyromagnetic_ratios.items():
        g = ratio * 1e6 * 4 * pi * units._mp / units._e
        print(f'":math:`^{{{n}}}`\\ {symbol}", {g:.3f}', file=fd)
