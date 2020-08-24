import sys
from typing import List, Tuple, Dict

import numpy as np
from ase.dft.dos import DOS, linear_tetrahedron_integration as lti
from ase.units import Ha

from gpaw import GPAW
from gpaw.utilities.dos import raw_orbital_LDOS, delta


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


class IBZWaveFunctions:
    def __init__(self, calc):
        self.calc = calc

    def weights(self):
        return self.calc.wfs.kd.weight_k

    def eigenvalues(self):
        kd = self.calc.wfs.kd
        eigs = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                          for k in range(kd.nibzkpts)]
                         for s in range(kd.nspins)])
        return eigs - self.calc.get_fermi_level()


def get_projector_numbers(setup, ell: int) -> List[int]:
    indices = []
    i1 = 0
    for n, l in zip(setup.n_j, setup.l_j):
        i2 = i1 + 2 * l + 1
        if l == ell and n >= 0:
            indices += list(range(i1, i2))
        i1 = i2
    return indices


def dos(filename,
        plot=False,
        output='dos.csv',
        width=0.1,
        integrated=False,
        projection=None,
        soc=False,
        emin=None,
        emax=None,
        npoints=400,
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
    calc = GPAW(filename, txt=None)
    # dos = DOS(calc, width, (emin, emax), npoints)
    wfs = IBZWaveFunctions(calc)

    nspins = calc.get_number_of_spins()
    spinlabels = [''] if nspins == 1 else [' up', ' dn']

    eigs = wfs.eigenvalues()
    weights = wfs.weights()

    emin = emin if emin is not None else eigs.min() - 0.1
    emax = emax if emax is not None else eigs.max() - 0.1

    energies = np.linspace(emin, emax, npoints)
    print(emin, emax, energies)

    if soc:
        eigs = eigs[np.newaxis]

    def dos(eigs, weigths):
        return _dos(eigs, weights, energies, width)

    D = []
    labels = []

    if projection is None or show_total:
        for eig, label in zip(eigs, spinlabels):
            D.append(dos(eig, weights))
            labels.append(label)

    if projection is not None:
        symbols = calc.atoms.get_chemical_symbols()
        indices, plabels = parse_projection_string(
            projection, symbols, calc.setups)
        weight_sknp = wfs.ldos(indices).transpose((2, 0, 1, 3))
        for label, pp in plabels.items():
            for eig, spinlabel, weight_knp in zip(eigs,
                                                  spinlabels,
                                                  weight_sknp):
                d = dos(eig, weight_knp[:, :, pp].sum(2))
                D.append(d)
                labels.append(label + spinlabel)

    if integrated:
        de = dos.energies[1] - dos.energies[0]
        dos.energies += de / 2
        D = [np.cumsum(d) * de for d in D]
        ylabel = 'iDOS [electrons]'
    else:
        ylabel = 'DOS [electrons/eV]'

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
        plt.ylabel(ylabel)
        plt.xlabel(r'$\epsilon-\epsilon_F$ [eV]')
        plt.show()


def parse_projection_string(projection, symbols, setups):
    p = 0
    indices: List[Tuple[int, List[int]]] = []
    plabels: Dict[str, List[int]] = {}
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
            ell = int(l)
            ii = get_projector_numbers(setups[A[0]], ell)
            label = s + '-' + l
            plabels[label] = []
            for a in A:
                indices.append((a, ii))
                plabels[label].append(p)
                p += 1

    return indices, plabels


def _dos(eigs, weights, energies, width):
    dos = 0.0
    for e, w in zip(eigs, weights):
        print(e.shape, w, energies.shape, width)
        dos += w * delta(energies, e, width)
    return dos


def _dos2():
    kd = calc.wfs.kd
    eigs.shape = (kd.nibzkpts, -1)
    eigs = eigs[kd.bz2ibz_k]
    eigs.shape = tuple(kd.N_c) + (-1,)
    weights.shape = (kd.nibzkpts, -1)
    weights /= kd.weight_k[:, np.newaxis]
    w = weights[kd.bz2ibz_k]
    w.shape = tuple(kd.N_c) + (-1,)
    dos = lti(calc.atoms.cell, eigs, energies, w)
    return dos
