from __future__ import annotations
from pathlib import Path

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
    parameter_functions[func.__name__] = func


def update_dict(default, value) -> dict[str, Any]:
    dct = default.copy()
    if value is not None:
        assert value.keys() <= default.keys(), (value, default)
        dct.update(value)
    return dct


class InputParameters:
    h: float | None
    parallel: dict[str, Any]
    txt: str | Path | IO[str] | None
    mode: dict[str, Any]
    xc: dict[str, Any]
    symmetry: dict[str, Any]
    kpts: dict[str, Any]
    setups: Any
    basis: Any
    magmoms: Any
    gpts: None | Sequence[int]
    charge: float
    nbands: None | int | float
    spinpol: bool

    def __init__(self, params: dict[str, Any]):
        """Accuracy of the self-consistency cycle."""
        self.params = params

        for key in params:
            if key not in parameter_functions:
                raise ValueError(
                    f'Unknown parameter {key!r}.  Must be one of: ' +
                    ', '.join(parameter_functions))
        for key, func in parameter_functions.items():
            if key in params:
                value = func(params[key])
            else:
                value = func()
            self.__dict__[key] = value

    def __repr__(self) -> str:
        p = ', '.join(f'{key}={value!r}'
                      for key, value in self.params.items())
        return f'InputParameters({p})'


@input_parameter
def occupations(value=None):
    return value


@input_parameter
def poissonsolver(value=None):
    """Poisson solver."""
    return value or {}


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
def eigensolver(value=None):
    """Eigensolver."""
    return value or {'converge': 'occupied'}


@input_parameter
def charge(value=0.0):
    return value


@input_parameter
def mixer(value=None):
    return value or {}


@input_parameter
def hund(value=False):
    """Using Hund's rule for guessing initial magnetic moments."""
    return value


@input_parameter
def xc(value='LDA'):
    """Exchange-Correlation functional."""
    if isinstance(value, str):
        return {'name': value}


@input_parameter
def mode(value='fd'):
    return {'name': value} if isinstance(value, str) else value


@input_parameter
def setups(value='paw'):
    """PAW datasets or pseudopotentials."""
    return value if isinstance(value, dict) else {None: value}


@input_parameter
def symmetry(value='undefined'):
    """Use of symmetry."""
    if value == 'undefined':
        value = {}
    elif value in {None, 'off'}:
        value = {'point_group': False, 'time_reversal': False}
    return value


@input_parameter
def basis(value=None):
    """Atomic basis set."""
    return value or {}


@input_parameter
def magmoms(value=None):
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
def maxiter(value=333):
    """Maximum number of SCF-iterations."""
    return value


@input_parameter
def h(value=None):
    """Grid spacing."""
    return value


@input_parameter
def txt(value: str | Path | IO[str] | None = '?'
        ) -> str | Path | IO[str] | None:
    """Log file."""
    return value


@input_parameter
def random(value=False):
    return value


@input_parameter
def spinpol(value=False):
    return value


@input_parameter
def gpts(value=None):
    """Number of grid points."""
    return value


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
def soc(value=False):
    return value


@input_parameter
def convergence(value=None):
    return update_dict({'energy': 0.0005,  # eV / electron
                        'density': 1.0e-4,  # electrons / electron
                        'eigenstates': 4.0e-8,  # eV^2 / electron
                        'forces': np.inf},
                       value)
