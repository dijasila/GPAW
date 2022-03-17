from __future__ import annotations
from pathlib import Path
import warnings

from typing import Any, IO, Sequence

import numpy as np
from gpaw.mpi import world

parameter_functions = {}

"""
background_charge
external
reuse_wfs_method
"""


def input_parameter(func):
    """Decorator for input-parameter normalization functions."""
    parameter_functions[func.__name__] = func
    return func


def update_dict(default: dict, value: dict | None) -> dict[str, Any]:
    """Create dict with defaults + updates.

    >>> update_dict({'a': 1, 'b': 'hello'}, {'a': 2})
    {'a': 2, 'b': 'hello'}
    >>> update_dict({'a': 1, 'b': 'hello'}, None)
    {'a': 1, 'b': 'hello'}
    >>> update_dict({'a': 1, 'b': 'hello'}, {'c': 2})
    Traceback (most recent call last):
    ValueError: Unknown key: 'c'. Must be one of a, b
    """
    dct = default.copy()
    if value is not None:
        if not (value.keys() <= default.keys()):
            key = (value.keys() - default.keys()).pop()
            raise ValueError(
                f'Unknown key: {key!r}. Must be one of {", ".join(default)}')
        dct.update(value)
    return dct


class InputParameters:
    basis: Any
    charge: float
    convergence: dict[str, Any]
    eigensolver: dict[str, Any]
    force_complex_dtype: bool
    gpts: None | Sequence[int]
    h: float | None
    kpts: dict[str, Any]
    magmoms: Any
    mode: dict[str, Any]
    nbands: None | int | float
    parallel: dict[str, Any]
    poissonsolver: dict[str, Any]
    setups: Any
    spinpol: bool
    symmetry: dict[str, Any]
    txt: str | Path | IO[str] | None
    xc: dict[str, Any]

    def __init__(self, params: dict[str, Any]):
        self.keys = set(params)

        for key in params:
            if key not in parameter_functions:
                raise ValueError(
                    f'Unknown parameter {key!r}.  Must be one of: ' +
                    ', '.join(parameter_functions))
        for key, func in parameter_functions.items():
            if key in params:
                param = params[key]
                if hasattr(param, 'todict'):
                    param = param.todict()
                value = func(param)
            else:
                value = func()
            self.__dict__[key] = value

        if self.h is not None and self.gpts is not None:
            raise ValueError("""You can't use both "gpts" and "h"!""")

        bands = self.convergence.pop('bands', None)
        if bands is not None:
            self.eigensolver['converge_bands'] = bands
            warnings.warn(f'Please use eigensolver={self.eigensolver!r}',
                          stacklevel=4)

        force_complex_dtype = self.mode.pop('force_complex_dtype', None)
        if force_complex_dtype is not None:
            warnings.warn(
                'Please use '
                f'GPAW(force_complex_dtype={bool(force_complex_dtype)}, ...)',
                stacklevel=3)
            self.force_complex_dtype = force_complex_dtype

    def __repr__(self) -> str:
        p = ', '.join(f'{key}={value!r}'
                      for key, value in self.items())
        return f'InputParameters({p})'

    def items(self):
        for key in self.keys:
            yield key, getattr(self, key)


@input_parameter
def basis(value=None):
    """Atomic basis set."""
    return value or {}


@input_parameter
def charge(value=0.0):
    return value


@input_parameter
def convergence(value=None):
    """Accuracy of the self-consistency cycle."""
    dct = update_dict({'energy': 0.0005,  # eV / electron
                       'density': 1.0e-4,  # electrons / electron
                       'eigenstates': 4.0e-8,  # eV^2 / electron
                       'forces': np.inf,
                       'bands': None,
                       'maximum iterations': None},
                      value)
    return {k: v for k, v in dct.items() if v is not None}


@input_parameter
def eigensolver(value=None) -> dict:
    """Eigensolver."""
    if isinstance(value, str):
        return {'name': value}
    return value or {}


@input_parameter
def force_complex_dtype(value: bool = False):
    return value


@input_parameter
def gpts(value=None):
    """Number of grid points."""
    return value


@input_parameter
def h(value=None):
    """Grid spacing."""
    return value


@input_parameter
def hund(value=False):
    """Using Hund's rule for guessing initial magnetic moments."""
    return value


@input_parameter
def kpts(value=None) -> dict[str, Any]:
    """Brillouin-zone sampling."""
    if value is None:
        value = {'size': (1, 1, 1)}
    elif not isinstance(value, dict):
        if len(value) == 3 and isinstance(value[0], int):
            value = {'size': value}
        else:
            value = {'points': np.array(value)}
    return value


@input_parameter
def magmoms(value=None):
    return value


@input_parameter
def maxiter(value=333):
    """Maximum number of SCF-iterations."""
    return value


@input_parameter
def mixer(value=None):
    return value or {}


@input_parameter
def mode(value='fd'):
    return {'name': value} if isinstance(value, str) else value


@input_parameter
def nbands(value: str | int | None = None) -> int | float | None:
    """Number of electronic bands."""
    if isinstance(value, int) or value is None:
        return value
    if nbands[-1] == '%':
        return float(value[:-1]) / 100
    raise ValueError('Integer expected: Only use a string '
                     'if giving a percentage of occupied bands')


@input_parameter
def occupations(value=None):
    return value


@input_parameter
def parallel(value: dict[str, Any] = None) -> dict[str, Any]:
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


@input_parameter
def poissonsolver(value=None):
    """Poisson solver."""
    return value or {}


@input_parameter
def random(value=False):
    return value


@input_parameter
def setups(value='paw'):
    """PAW datasets or pseudopotentials."""
    return value if isinstance(value, dict) else {None: value}


@input_parameter
def soc(value=False):
    return value


@input_parameter
def spinpol(value=False):
    return value


@input_parameter
def symmetry(value='undefined'):
    """Use of symmetry."""
    if value == 'undefined':
        value = {}
    elif value in {None, 'off'}:
        value = {'point_group': False, 'time_reversal': False}
    return value


@input_parameter
def txt(value: str | Path | IO[str] | None = '?'
        ) -> str | Path | IO[str] | None:
    """Log file."""
    return value


@input_parameter
def xc(value='LDA'):
    """Exchange-Correlation functional."""
    if isinstance(value, str):
        return {'name': value}
    return value
