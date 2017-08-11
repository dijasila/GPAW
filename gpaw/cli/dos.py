from __future__ import print_function
import sys

import numpy as np
from ase.dft.dos import DOS

from gpaw import GPAW


def dos(filename, plot=False, output='dos.csv', width=0.1, integrated=False):
    """Calculate density of states.

    filename: str
        Name of restart-file.
    plot: bool
        Show a plot.
    output: str
        Name of CSV output file.
    width: float
        Width of Gaussians.
    integrated: bool
        Calculate integratede DOS.
    """
    calc = GPAW(filename, txt=None)
    dos = DOS(calc, width)
    D = [dos.get_dos(spin) for spin in range(calc.get_number_of_spins())]
    if integrated:
        de = dos.energies[1] - dos.energies[0]
        dos.energies += de / 2
        D = [np.cumsum(d) * de for d in D]

    if output:
        fd = sys.stdout if output == '-' else open(output, 'w')
        for x in zip(dos.energies, *D):
            print(*x, sep=', ', file=fd)
        if output != '-':
            fd.close()
    if plot:
        import matplotlib.pyplot as plt
        for y in D:
            plt.plot(dos.energies, y)
        plt.show()


class CLICommand:
    short_description = 'Calculate density of states from gpw-file'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')
        parser.add_argument('csv', nargs='?', metavar='csv-file')
        parser.add_argument('-p', '--plot', action='store_true')
        parser.add_argument('-i', '--integrated', action='store_true')
        parser.add_argument('-w', '--width', type=float, default=0.1)

    @staticmethod
    def run(args):
        dos(args.gpw, args.plot, args.csv, args.width, args.integrated)
