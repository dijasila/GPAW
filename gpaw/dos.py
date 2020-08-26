from pathlib import Path
from typing import Union, List, Any

import numpy as np
from ase.spectrum.dosdata import GridDOSData

from gpaw import GPAW
from ase.dft.dos import linear_tetrahedron_integration as lti
from gpaw.setup import Setup
from gpaw.spinorbit import soc_eigenstates, BZWaveFunctions

Array1D = Any
Array3D = Any


class IBZWaveFunctions:
    """Container for eigenvalues and PAW projections (only IBZ)."""
    def __init__(self, calc: GPAW):
        self.calc = calc
        self.fermi_level = self.calc.get_fermi_level()
        self.size = calc.wfs.kd.N_c
        self.bz2ibz_map = calc.wfs.kd.bz2ibz_k

    def weights(self) -> Array1D:
        """Weigths of IBZ k-points (adds to 1.0)."""
        return self.calc.wfs.kd.weight_k

    def eigenvalues(self) -> Array3D:
        """All eigenvalues."""
        kd = self.calc.wfs.kd
        eigs = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                          for k in range(kd.nibzkpts)]
                         for s in range(kd.nspins)])
        return eigs

    def pdos_weights(self,
                     a: int,
                     indices: List[int]
                     ) -> Array3D:
        """Projections for PDOS.

        Returns (nibzkpts, nbands, nspins, nindices)-shaped ndarray
        of the square of absolute value of the projections.  The *indices*
        list contains (atom-number, projector-numbers) tuples.
        """
        kd = self.calc.wfs.kd
        dos_kns = np.zeros((kd.nibzkpts,
                            self.calc.wfs.bd.nbands,
                            kd.nspins))
        bands = self.calc.wfs.bd.get_slice()

        for wf in self.calc.wfs.kpt_u:
            P_ani = wf.projections
            if a in P_ani:
                P_ni = P_ani[a][:, indices]
                dos_kns[wf.k, bands, wf.s] = (abs(P_ni)**2).sum(1)

        self.calc.world.sum(dos_kns)
        return dos_kns


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


def gaussian_dos(eig_kn,
                 weight_kn,
                 weight_k,
                 energies,
                 width: float) -> Array1D:
    """Simple broadening with a Gaussian."""
    dos = np.zeros_like(energies)
    if weight_kn is None:
        for e_n, w in zip(eig_kn, weight_k):
            for e in e_n:
                dos += w * np.exp(-((energies - e) / width)**2)
    else:
        for e_n, w, w_n in zip(eig_kn, weight_k, weight_kn):
            for e, w2 in zip(e_n, w_n):
                dos += w * w2 * np.exp(-((energies - e) / width)**2)
    return dos / (np.pi**0.5 * width)


def linear_tetrahedron_dos(eig_kn,
                           weight_kn,
                           energies,
                           cell,
                           size,
                           bz2ibz_map=None) -> Array1D:
    """Linear-tetrahedron method."""
    if len(eig_kn) != np.prod(size):
        eig_kn = eig_kn[bz2ibz_map]
        if weight_kn is not None:
            weight_kn = weight_kn[bz2ibz_map]

    shape = tuple(size) + (-1,)
    eig_kn = eig_kn.reshape(shape)
    if weight_kn is not None:
        weight_kn = weight_kn.reshape(shape)

    dos = lti(cell, eig_kn, energies, weight_kn)
    return dos


class DOSCalculator:
    def __init__(self,
                 wfs,
                 emin=None,
                 emax=None,
                 npoints=200,
                 setups=None,
                 cell=None):
        self.wfs = wfs
        self.setups = setups
        self.cell = cell

        self.eig_skn = wfs.eigenvalues() - wfs.fermi_level
        if self.eig_skn.ndim == 2:
            self.eig_skn = self.eig_skn[np.newaxis]

        emin = self.eig_skn.min() - 0.5 if emin is None else emin
        emax = self.eig_skn.max() + 0.5 if emax is None else emax
        self.energies = np.linspace(emin, emax, npoints)

        self.nspins = len(self.eig_skn)
        self.weight_k = wfs.weights()

        self.cache = {}

    @classmethod
    def from_calculator(cls,
                        filename: Union[GPAW, Path, str],
                        emin=None, emax=None, npoints=200,
                        soc=False):
        """

        filename: str
            Name of restart-file.
        """
        if isinstance(filename, GPAW):
            calc = filename
        else:
            calc = GPAW(filename, txt=None)

        wfs: Union[BZWaveFunctions, IBZWaveFunctions]
        if soc:
            wfs = soc_eigenstates(calc)
        else:
            wfs = IBZWaveFunctions(calc)

        return DOSCalculator(wfs, emin, emax, npoints,
                             calc.setups, calc.atoms.cell)

    def calculate(self, eig_kn, weight_kn=None, width=0.1):
        if width > 0.0:
            return gaussian_dos(eig_kn, weight_kn,
                                self.weight_k, self.energies, width)
        else:
            return linear_tetrahedron_dos(
                eig_kn, weight_kn, self.energies,
                self.cell, self.wfs.size, self.wfs.bz2ibz_map)

    def dos(self,
            spin=None,
            width=0.1):
        """Calculate density of states.

        width: float
            Width of Gaussians in eV. LT.......
        """
        if spin is None:
            dos = sum(self.calculate(eig_kn, width=width)
                      for eig_kn in self.eig_skn)
            dos *= 2 / self.nspins
            label = 'DOS'
        else:
            dos = self.calculate(self.eig_skn[spin], width=width)
            label = 'DOS ({})'.format('up' if spin == 0 else 'dn')

        return GridDOSData(self.energies, dos, {'label': label})

    def pdos(self, a, l, spin=None, width=0.1):
        if (a, l) in self.cache:
            weight_kns = self.cache[(a, l)]
        else:
            indices = get_projector_numbers(self.setups[a], l)
            weight_kns = self.wfs.pdos_weights(a, indices)
            self.cache.clear()
            self.cache[(a, l)] = weight_kns

        label = 'atom #{}-{}'.format(a, 'spdfg'[l])

        if spin is None:
            dos = sum(self.calculate(eig_kn, weight_nk.T, width=width)
                      for eig_kn, weight_nk in zip(self.eig_skn, weight_kns.T))
            dos *= 2 / self.nspins
        else:
            dos = self.calculate(self.eig_skn[spin], weight_kns[:, :, spin],
                                 width=width)
            label += ' ({})'.format('up' if spin == 0 else 'dn')

        return GridDOSData(self.energies, dos, {'label': label})
