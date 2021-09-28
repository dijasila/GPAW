from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import IO

import numpy as np
from gpaw.mpi import world
from gpaw.scf import dict2criterion



default_parameters = {
    'soc': None,
    'background_charge': None,
    'external': None,
    'occupations': None,
    'mixer': None,
    'reuse_wfs_method': 'paw',
    'maxiter': 333,
    'convergence': {
        'energy': 0.0005,  # eV / electron
        'density': 1.0e-4,  # electrons / electron
        'eigenstates': 4.0e-8,  # eV^2 / electron
        'forces': np.inf}}


def update_dict(default, value) -> dict[str, Any]:
    dct = default.copy()
    if value is not None:
        assert value.keys() < default.keys()
        dct.update(value)
    return dct


class InputParameters:
    def __init__(self, params):
        """Accuracy of the self-consistency cycle."""
        self.params = params

        self.parallel = parallel(params.get('parallel'))
        self.convergence = convergence(params.get('convergence')
parallel, mode, xc, basis, setups, kpts, h, gpts, symmetry, charge, magmoms
background_charge': None,
external': None,

hund
random
nbands

occupations': None,
mixer': None,
reuse_wfs_method': 'paw',
maxiter': 333}
convergence
poissonsolver
eigensolver

        self.poissonsolver(value=None):
    """Poisson solver."""
    if value is None:
        value = {}
    return value


def parallel(value: dict[str, Any]) -> dict[str, Any]:
    dct = update_dict({'kpt': None,
                       'domain': None,
                       'band': None,
                       'order': 'kdb',
                       'stridebands': False,
                       'augment_grids': False,
                       'sl_auto': False,
                       'sl_default': None,
                       'sl_diagonalize': None,
                       'sl_inverse_cholesky': None,
                       'sl_lcao': None,
                       'sl_lrtddft': None,
                       'use_elpa': False,
                       'elpasolver': '2stage',
                       'buffer_size': None,
                       'world': None},
                      value)
    dct['world'] = dct['world'] or world
    return dct


def eigensolver(value=None):
    """Eigensolver."""
    return value or {'converge': 'occupied'}


def charge(value=0.0):
    return value


def hund(value=False):
    """Using Hund's rule for guessing initial magnetic moments."""
    return value


def xc(value='LDA'):
    """Exchange-Correlation functional."""
    if isinstance(value, str):
        return {'name': value}


def mode(value='fd'):
    return {'name': value} if isinstance(value, str) else value


def setups(atomic_numbers, basis, xc_name, world, value='paw'):
    """PAW datasets or pseudopotentials."""
    from gpaw.setup import Setups
    return Setups(atomic_numbers,
                  value,
                  basis,
                  xc_name,
                  world)


@input_parameter
def symmetry(atoms, setups, magmoms, value=None):
    """Use of symmetry."""
    from gpaw.new.symmetry import Symmetry
    from gpaw.symmetry import Symmetry as OldSymmetry
    if value in {None, 'off'}:
        value = {}
    if magmoms is None:
        ids = setups.id_a
    elif magmoms.ndim == 1:
        ids = [id + (m,) for id, m in zip(setups.id_a, magmoms)]
    else:
        ids = [id + tuple(m) for id, m in zip(setups.id_a, magmoms)]
    symmetry = OldSymmetry(ids, atoms.cell, atoms.pbc, **value)
    symmetry.analyze(atoms.get_scaled_positions())
    return Symmetry(symmetry)


@input_parameter
def basis(value=None):
    """Atomic basis set."""
    return value or {}


@input_parameter
def magmoms(atoms, value=None):
    if value is None:
        magmoms = atoms.get_initial_magnetic_moments()
    elif isinstance(value, float):
        magmoms = np.zeros(len(atoms)) + value
    else:
        magmoms = np.array(value)

    collinear = magmoms.ndim == 1
    if collinear and not magmoms.any():
        magmoms = None

    return magmoms


@input_parameter
def kpts(atoms, value=None):
    """Brillouin-zone sampling."""
    from gpaw.new.brillouin import BZ, MonkhorstPackKPoints
    if value is None:
        value = {'size': (1, 1, 1)}
    elif not isinstance(value, dict):
        if len(value) == 3 and isinstance(value[0], int):
            value = {'size': value}
        else:
            value = {'points': np.array(value)}

    if 'points' in value:
        return BZ(value['points'])
    return MonkhorstPackKPoints(value['size'])


@input_parameter
def h(value=None):
    """Grid spacing."""
    return value


@input_parameter
def random(value=False):
    return value


@input_parameter
def gpts(value=None):
    """Number of grid points."""
    return value


@input_parameter
def nbands(setups,
           charge=0.0,
           magmoms=None,
           is_lcao: bool = False,
           value: str | int = None):
    """Number of electronic bands."""
    nbands = value
    nao = setups.nao
    nvalence = setups.nvalence - charge
    M = 0 if magmoms is None else np.linalg.norm(magmoms.sum(0))

    orbital_free = any(setup.orbital_free for setup in setups)
    if orbital_free:
        nbands = 1

    if isinstance(nbands, str):
        if nbands == 'nao':
            nbands = nao
        elif nbands[-1] == '%':
            cfgbands = (nvalence + M) / 2
            nbands = int(np.ceil(float(nbands[:-1]) / 100 * cfgbands))
        else:
            raise ValueError('Integer expected: Only use a string '
                             'if giving a percentage of occupied bands')

    if nbands is None:
        # Number of bound partial waves:
        nbandsmax = sum(setup.get_default_nbands()
                        for setup in setups)
        nbands = int(np.ceil((1.2 * (nvalence + M) / 2))) + 4
        if nbands > nbandsmax:
            nbands = nbandsmax
        if is_lcao and nbands > nao:
            nbands = nao
    elif nbands <= 0:
        nbands = max(1, int(nvalence + M + 0.5) // 2 + (-nbands))

    if nbands > nao and is_lcao:
        raise ValueError('Too many bands for LCAO calculation: '
                         '%d bands and only %d atomic orbitals!' %
                         (nbands, nao))

    if nvalence < 0:
        raise ValueError(
            'Charge %f is not possible - not enough valence electrons' %
            charge)

    if nvalence > 2 * nbands and not orbital_free:
        raise ValueError('Too few bands!  Electrons: %f, bands: %d'
                         % (nvalence, nbands))

    return nbands


