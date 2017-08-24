from __future__ import print_function
import sys

import numpy as np
from ase.dft.dos import DOS, ltidos
from ase.units import Ha

from gpaw import GPAW
from gpaw.utilities.dos import raw_orbital_LDOS, delta


class CLICommand:
    short_description = 'Calculate (projected) density of states from gpw-file'

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('gpw', metavar='gpw-file')
        add('csv', nargs='?', metavar='csv-file')
        add('-p', '--plot', action='store_true',
            help='Plot the DOS.')
        add('-i', '--integrated', action='store_true',
            help='Calculate integrated DOS.')
        add('-w', '--width', type=float, default=0.1,
            help='Width of Gaussian.  Use 0.0 to use linear tetrahedron '
            'interpolation.')
        add('-a', '--atom', help='Project onto atoms: "Cu-spd,H-s" or use '
            'atom indices "12-spdf".')
        add('-r', '--range', default=':', metavar='e1:e2',
            help='Energy range.  Examples: " -2:2", ":10.0", " -5:". '
            'Note that for a negative start energy, you need to use quotes '
            'and prepend a space!')
        add('-n', '--points', type=int, default=400, help='Number of points.')

    @staticmethod
    def run(args):
        emin, _, emax = args.range.partition(':')
        emin = float(emin) if emin else None
        emax = float(emax) if emax else None
        dos(args.gpw, args.plot, args.csv, args.width, args.integrated,
            args.atom, emin, emax, args.points)


def dos(filename, plot=False, output='dos.csv', width=0.1, integrated=False,
        projection=None, emin=None, emax=None, npoints=400):
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
    dos = DOS(calc, width, (emin, emax), npoints)

    nspins = calc.get_number_of_spins()
    spinlabels = [''] if nspins == 1 else [' up', ' dn']

    if projection is None:
        D = [dos.get_dos(spin) for spin in range(nspins)]
        labels = ['DOS' + sl for sl in spinlabels]
    else:
        D = []
        labels = []
        for p in projection.split(','):
            s, ll = p.split('-')
            if s.isdigit():
                A = [int(s)]
                s = '#' + s
            else:
                A = [a for a, symbol in
                     enumerate(calc.atoms.get_chemical_symbols())
                     if symbol == s]
            for spin in range(nspins):
                for l in ll:
                    d = 0.0
                    for a in A:
                        d += ldos(calc, a, spin, l, width, dos.energies)
                    labels.append(s + '-' + l + spinlabels[spin])
                    D.append(d)

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
        for y, label in zip(D, labels):
            plt.plot(dos.energies, y, label=label)
        plt.legend()
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
    return dos
