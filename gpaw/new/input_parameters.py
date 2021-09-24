from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import IO

import numpy as np
from gpaw.mpi import world
from gpaw.scf import dict2criterion

parameter_functions = {}


class InputParameters:
    def __init__(self, params, functions=None):
        self.params = params
        self.functions = functions or parameter_functions

    def __getattr__(self, name):
        param = self.params.get(name)
        if param is not None:
            if callable(param):
                return param
            func = self.functions[name]
            if func.__code__.co_argcount == 1:
                return func(param)
            else:
                return functools.partial(func, value=param)
        func = self.functions[name]
        if func.__code__.co_argcount == 1:
            return func()
        return func


default_parameters = {
    'soc': None,
    'background_charge': None,
    'external': None,

    'occupations': None,
    'mixer': None,
    'reuse_wfs_method': 'paw',
    'maxiter': 333}


def input_parameter(func):
    parameter_functions[func.__name__] = func
    return func


@input_parameter
def txt(read_from_file: bool, value: str | Path | IO[str] | None = '?'):
    """Where to send text output."""
    if value == '?':
        if read_from_file:
            value = None
        else:
            value = '-'
    return value


@input_parameter
def convergence(value=None):
    """Accuracy of the self-consistency cycle."""
    defaults = {
        'energy': 0.0005,  # eV / electron
        'density': 1.0e-4,  # electrons / electron
        'eigenstates': 4.0e-8,  # eV^2 / electron
        'forces': np.inf}
    value = value or {}
    assert value.keys() < defaults.keys()
    criteria = defaults.copy()
    criteria.update(value)

    # Gather convergence criteria for SCF loop.
    custom = criteria.pop('custom', [])
    for name, criterion in criteria.items():
        if hasattr(criterion, 'todict'):
            # 'Copy' so no two calculators share an instance.
            criteria[name] = dict2criterion(criterion.todict())
        else:
            criteria[name] = dict2criterion({name: criterion})

    if not isinstance(custom, (list, tuple)):
        custom = [custom]
    for criterion in custom:
        if isinstance(criterion, dict):  # from .gpw file
            msg = ('Custom convergence criterion "{:s}" encountered, '
                   'which GPAW does not know how to load. This '
                   'criterion is NOT enabled; you may want to manually'
                   ' set it.'.format(criterion['name']))
            warnings.warn(msg)
            continue

        criteria[criterion.name] = criterion
        msg = ('Custom convergence criterion {:s} encountered. '
               'Please be sure that each calculator is fed a '
               'unique instance of this criterion. '
               'Note that if you save the calculator instance to '
               'a .gpw file you may not be able to re-open it. '
               .format(criterion.name))
        warnings.warn(msg)

    for criterion in criteria.values():
        criterion.reset()

    return criteria


@input_parameter
def poissonsolver(value=None):
    """Poisson solver."""
    if value is None:
        value = {}
    return value


@input_parameter
def eigensolver(value=None):
    """Eigensolver."""
    defaults = {
        'converge': 'occupied'}
    value = value or {}
    assert value.keys() < defaults.keys()
    dct = defaults.copy()
    dct.update(value)
    return dct


@input_parameter
def charge(value=0.0):
    return value


@input_parameter
def hund(value=False):
    """Using Hund's rule for guessing initial magnetic moments."""
    return value


@input_parameter
def xc(value='LDA'):
    """Exchange-Correlation functional."""
    from gpaw.new.xc import XCFunctional
    from gpaw.xc import XC

    if isinstance(value, str):
        value = {'name': value}
    return XCFunctional(XC(value))


@input_parameter
def mode(value='fd'):
    from gpaw.new.modes import FDMode
    return FDMode()


@input_parameter
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


@input_parameter
def parallel(value=None):
    defaults = {
        'kpt': None,
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
        'world': None}
    value = value or {}
    assert value.keys() < defaults.keys()
    dct = defaults.copy()
    dct.update(value)
    dct['world'] = dct['world'] or world
    return dct
