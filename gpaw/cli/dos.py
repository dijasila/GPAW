from __future__ import print_function
import sys

import numpy as np
from ase.dft.dos import DOS, ltidos
from ase.units import Ha

from gpaw import GPAW
from gpaw.utilities.dos import raw_orbital_LDOS, delta


def dos(filename, plot=False, output='dos.csv', width=0.1, integrated=False,
        projection=None):
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
        Calculate integrated DOS.

    """
    calc = GPAW(filename, txt=None)
    dos = DOS(calc, width)
    if projection is None:
        D = [dos.get_dos(spin) for spin in range(calc.get_number_of_spins())]
    else:
        for p in projection.split(','):
            a, ll = p.split('-')
            if a.isnumber():
                A = [int(a)]
            else:
                A = [a for a, symbol in calc.atoms.symbols if symbol == a]
            for spin in range(calc.get_number_of_spins()):
                for l in ll:
                    dos = 0.0
                    for a in A:
                        dos += ldos(calc, a, spin, l, width)
                    D.append(dos)

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


def ldos(calc, a, spin, l, width, energies):
    eigs, weights = raw_orbital_LDOS(calc, a, spin, l)
    eigs *= Ha
    if width > 0.0:
        dos = 0.0
        for e, w in zip(eigs, weights):
            dos += w * delta(energies, e, width)
    else:
        kd = calc.wfs.kd
        eigs.shape = (kd.nibzkpts, -1)
        eigs = eigs[kd.bz2ibz_k]
        eigs.shape = tuple(kd.N_c) + (-1,)
        weights.shape = (kd.nibzkpts, -1)
        weights /= kd.weight_k[:, np.newaxis]
        w = weights[kd.bz2ibz_k]
        w.shape = tuple(kd.N_c) + (-1,)
        dos = ltidos(calc.atoms.cell, eigs, energies, w)


class CLICommand:
    short_description = 'Calculate density of states from gpw-file'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')
        parser.add_argument('csv', nargs='?', metavar='csv-file')
        parser.add_argument('-p', '--plot', action='store_true')
        parser.add_argument('-i', '--integrated', action='store_true')
        parser.add_argument('-w', '--width', type=float, default=0.1)
        parser.add_argument('-a', '--atom')

    @staticmethod
    def run(args):
        dos(args.gpw, args.plot, args.csv, args.width, args.integrated,
            args.atom)
