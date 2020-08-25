"""CLI-code for dos-subcommand."""
from pathlib import Path
from typing import Any, Union, List, Tuple

import numpy as np
from ase.spectrum.dosdata import GridDOSData
from ase.spectrum.doscollection import GridDOSCollection

from gpaw import GPAW
from gpaw.setup import Setup
from gpaw.dos import DOSCalculator


class CLICommand:
    """Calculate (projected) density of states from gpw-file."""

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
        add('-t', '--total', action='store_true',
            help='Show both PDOS and total DOS.')
        add('-r', '--range', nargs=2, metavar=('emin', 'emax'),
            help='Energy range in eV relative to Fermi level.')
        add('-n', '--points', type=int, default=400, help='Number of points.')
        add('--soc', action='store_true',
            help='Include spin-orbit coupling.')

    @staticmethod
    def run(args):
        if args.range is None:
            emin = None
            emax = None
        else:
            emin, emax = (float(e) for e in args.range)
        dos(args.gpw, args.plot, args.csv, args.width, args.integrated,
            args.atom,
            args.soc,
            emin, emax, args.points, args.total)


Array1D = Any
Array3D = Any


def parse_projection_string(projection: str,
                            symbols: List[str],
                            setups: List[Setup]
                            ) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Don't ask.

    >>> from gpaw.setup import create_setup
    >>> setup = create_setup('Li')
    >>> a, b = parse_projection_string('Li-sp', ['Li', 'Li'], [setup, setup])
    >>> a
    [(0, [0]), (1, [0]), (0, [1, 2, 3]), (1, [1, 2, 3])]
    >>> b
    {'Li-s': [0, 1], 'Li-p': [2, 3]}
    """
    result: List[Tuple[str, List[Tuple[int, int]]]] = []
    for proj in projection.split(','):
        s, ll = proj.split('-')
        if s.isdigit():
            A = [int(s)]
            s = '#' + s
        else:
            A = [a for a, symbol in
                 enumerate(symbols)
                 if symbol == s]
            if not A:
                raise ValueError('No such atom: ' + s)
        for l in ll:
            ell = 'spdf'.index(l)
            label = s + '-' + l
            result.append((label, [(a, ell) for a in A]))

    return result


def dos(filename: Union[Path, str],
        plot=False,
        output='dos.csv',
        width=0.1,
        integrated=False,
        projection=None,
        soc=False,
        emin=None,
        emax=None,
        npoints=200,
        show_total=None):
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
    calc = GPAW(filename)

    doscalc = DOSCalculator.from_calculator(calc, emin, emax, npoints, soc)

    energies = doscalc.energies
    nspins = doscalc.nspins
    spinlabels = [''] if nspins == 1 else [' up', ' dn']
    spins = [None] if nspins == 1 else [0, 1]

    dosobjs = GridDOSCollection([], energies)

    if projection is None or show_total:
        for spin, label in zip(spins, spinlabels):
            dosobjs += doscalc.dos(spin=spin, width=width)

    if projection is not None:
        symbols = calc.atoms.get_chemical_symbols()
        projs = parse_projection_string(
            projection, symbols, calc.setups)
        for label, contributions in projs:
            for spin, spinlabel in zip(spins, spinlabels):
                dos = np.zeros_like(energies)
                for a, indices in contributions:
                    obj = doscalc.pdos(a, indices, spin=spin, width=width)
                    dos += obj.get_weights()
                dosobjs += GridDOSData(energies, dos,
                                       {'label': label + spinlabel})

    if integrated:
        de = energies[1] - energies[0]
        energies = energies + de / 2
        dosobjs = GridDOSCollection(
            [GridDOSData(energies,
                         np.cumsum(obj.get_weights()) * de,
                         obj.info)
             for obj in dosobjs])
        ylabel = 'iDOS [electrons]'
    else:
        ylabel = 'DOS [electrons/eV]'

    if plot:
        ax = dosobjs.plot()
        ax.set_xlabel(r'$\epsilon-\epsilon_F$ [eV]')
        ax.set_ylabel(ylabel)
        import matplotlib.pyplot as plt
        plt.show()
