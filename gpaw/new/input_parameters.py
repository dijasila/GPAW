from __future__ import annotations
import numpy as np
import functools
from gpaw.mpi import world

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
    'eigensolver': None,
    'reuse_wfs_method': 'paw',
    'maxiter': 333,
    'idiotproof': True,
    'convergence': {'energy': 0.0005,  # eV / electron
                    'density': 1.0e-4,  # electrons / electron
                    'eigenstates': 4.0e-8,  # eV^2 / electron
                    'bands': 'occupied'},
    'verbose': 0}  # deprecated


def input_parameter(func):
    parameter_functions[func.__name__] = func
    return func


@input_parameter
def poissonsolver(value=None):
    if value is None:
        value = {}
    return value


@input_parameter
def charge(value=0.0):
    return value


@input_parameter
def hund(value=False):
    return value


@input_parameter
def xc(value='LDA'):
    from gpaw.xc import XC
    from gpaw.new.xc import XCFunctional

    if isinstance(value, str):
        value = {'name': value}
    return XCFunctional(XC(value))


@input_parameter
def mode(value='fd'):
    from gpaw.new.modes import FDMode
    return FDMode()


@input_parameter
def setups(atomic_numbers, basis, xc_name, world, value='paw'):
    from gpaw.setup import Setups
    return Setups(atomic_numbers,
                  value,
                  basis,
                  xc_name,
                  world)


@input_parameter
def symmetry(atoms, setups, magmoms, value=None):
    from gpaw.symmetry import Symmetry as OldSymmetry
    from gpaw.new.symmetry import Symmetry
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
    return value


@input_parameter
def random(value=False):
    return value


@input_parameter
def gpts(value=None):
    return value


@input_parameter
def nbands(setups,
           charge=0.0,
           magmoms=None,
           is_lcao: bool = False,
           value: str | int = None):
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
            basebands = (nvalence + M) / 2
            nbands = int(np.ceil(float(nbands[:-1]) / 100 * basebands))
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
