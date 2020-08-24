from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any, Union

import numpy as np
from ase.dft.dos import linear_tetrahedron_integration as lti

from gpaw import GPAW
from gpaw.setup import Setup
from gpaw.spinorbit import soc_eigenstates, BZWaveFunctions


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


class IBZWaveFunctions:
    def __init__(self, calc: GPAW):
        self.calc = calc
        self.fermi_level = self.calc.get_fermi_level()

    def weights(self) -> Array1D:
        return self.calc.wfs.kd.weight_k

    def eigenvalues(self) -> Array3D:
        kd = self.calc.wfs.kd
        eigs = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                          for k in range(kd.nibzkpts)]
                         for s in range(kd.nspins)])
        return eigs

    def ldos(self,
             indices: List[Tuple[int, List[int]]]
             ) -> Array3D:

        kd = self.calc.wfs.kd
        dos_knsp = np.zeros((kd.nibzkpts,
                             self.calc.wfs.bd.nbands,
                             2,
                             len(indices)))
        bands = self.calc.wfs.bd.get_slice()

        for wf in self.calc.wfs.kpt_u:
            P_ani = wf.projections
            p = 0
            for a, ii in indices:
                if a in P_ani:
                    P_ni = P_ani[a][:, ii]
                    dos_knsp[wf.k, bands, wf.s, p] = (abs(P_ni)**2).sum(1)
                p += 1

        self.calc.world.sum(dos_knsp)
        return dos_knsp


def get_projector_numbers(setup: Setup, ell: int) -> List[int]:
    """Find indices of bound-state PAW projector functions.

    >>> from gpaw.setup import create_setup
    >>> setup = create_setup('Li')
    >>> get_projector_numbers(setup, 0)
    [0]
    >>> get_projector_numbers(setup, 1)
    [1, 2, 3]
    """
    indices = []
    i1 = 0
    for n, l in zip(setup.n_j, setup.l_j):
        i2 = i1 + 2 * l + 1
        if l == ell and n >= 0:
            indices += list(range(i1, i2))
        i1 = i2
    return indices


def dos(filename: Union[Path, str],
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

    wfs: Union[BZWaveFunctions, IBZWaveFunctions]
    if soc:
        wfs = soc_eigenstates(calc)
    else:
        wfs = IBZWaveFunctions(calc)

    nspins = calc.get_number_of_spins()
    spinlabels = [''] if nspins == 1 else [' up', ' dn']

    eig_skn = wfs.eigenvalues() - wfs.fermi_level
    if soc:
        eig_skn = eig_skn[np.newaxis]

    emin = emin if emin is not None else eig_skn.min() - 0.1
    emax = emax if emax is not None else eig_skn.max() - 0.1

    energies = np.linspace(emin, emax, npoints)

    weight_k = wfs.weights()
    weight_kn = np.empty_like(eig_skn[0])
    weight_kn[:] = weight_k[:, np.newaxis]

    if width > 0.0:
        def dos(eig_kn, weight_kn):
            return _dos1(eig_kn, weight_kn, energies, width)
    else:
        def dos(eig_kn, weight_kn):
            return _dos2(eig_kn, weight_kn,
                         energies, calc.atoms.cell, calc.wfs.kd)

    D = []
    labels = []

    if projection is None or show_total:
        for eig_kn, label in zip(eig_skn, spinlabels):
            D.append(dos(eig_kn, weight_kn))
            labels.append('DOS' + label)

    if projection is not None:
        symbols = calc.atoms.get_chemical_symbols()
        indices, plabels = parse_projection_string(
            projection, symbols, calc.setups)
        weight_sknp = wfs.ldos(indices).transpose((2, 0, 1, 3))
        for label, pp in plabels.items():
            for eig_kn, spinlabel, weight_knp in zip(eig_skn,
                                                     spinlabels,
                                                     weight_sknp):
                d = dos(eig_kn, weight_knp[:, :, pp].sum(2) * weight_kn)
                D.append(d)
                labels.append(label + spinlabel)

    if integrated:
        de = energies[1] - energies[0]
        energies += de / 2
        D = [np.cumsum(d) * de for d in D]
        ylabel = 'iDOS [electrons]'
    else:
        ylabel = 'DOS [electrons/eV]'

    if output:
        fd = sys.stdout if output == '-' else open(output, 'w')
        for x in zip(energies, *D):
            print(*x, sep=', ', file=fd)
        if output != '-':
            fd.close()
    if plot:
        import matplotlib.pyplot as plt
        for y, label in zip(D, labels):
            plt.plot(energies, y, label=label)
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel(r'$\epsilon-\epsilon_F$ [eV]')
        plt.show()


def parse_projection_string(projection: str,
                            symbols: List[str],
                            setups: List[Setup]
                            ) -> Tuple[List[Tuple[int, List[int]]],
                                       Dict[str, List[int]]]:
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
            ell = 'spdf'.index(l)
            ii = get_projector_numbers(setups[A[0]], ell)
            label = s + '-' + l
            plabels[label] = []
            for a in A:
                indices.append((a, ii))
                plabels[label].append(p)
                p += 1

    return indices, plabels


def _dos1(eig_kn, weight_kn, energies, width):
    dos = 0.0
    for e_n, w_n in zip(eig_kn, weight_kn):
        for e, w in zip(e_n, w_n):
            dos += w * np.exp(-((energies - e) / width)**2)
    return dos / (np.pi**0.5 * width)


def _dos2(eig_kn, weight_kn, energies, cell, kd):
    if len(eig_kn) != kd.nbzkpts:
        assert len(eig_kn) == kd.nibzkpts
        eig_kn = eig_kn[kd.bz2ibz_k]
        weight_kn = weight_kn[kd.bz2ibz_k]

    weight_kn *= kd.nbzkpts

    eig_kn.shape = tuple(kd.N_c) + (-1,)
    weight_kn.shape = tuple(kd.N_c) + (-1,)

    dos = lti(cell, eig_kn, energies, weight_kn)
    return dos
