import numpy as np
import functools


default_parameters = {
    'h': None,  # Angstrom
    'gpts': None,
    'kpts': [(0.0, 0.0, 0.0)],
    'nbands': None,
    'charge': 0,
    'magmoms': None,
    'soc': None,
    'background_charge': None,
    'setups': {},
    'basis': {},
    'spinpol': None,
    'xc': 'LDA',

    'occupations': None,
    'poissonsolver': None,
    'mixer': None,
    'eigensolver': None,
    'reuse_wfs_method': 'paw',
    'external': None,
    'random': False,
    'hund': False,
    'maxiter': 333,
    'idiotproof': True,
    'convergence': {'energy': 0.0005,  # eV / electron
                    'density': 1.0e-4,  # electrons / electron
                    'eigenstates': 4.0e-8,  # eV^2 / electron
                    'bands': 'occupied'},
    'verbose': 0}  # deprecated


class InputParameter:
    def __init__(self, value=None):
        self.value = value


class Charge(InputParameter):
    def __init__(self, value=0.0):
        self.value = value


class Hund(InputParameter):
    def __init__(self, value=False):
        self.value = value


class XC(InputParameter):
    def __init__(self, value='LDA'):
        if isinstance(value, str):
            value = {'name': value}
        self.value = value

    def __call__(self):
        from gpaw.xc import XC
        return XC(self.value)


class Mode(InputParameter):
    def __init__(self, value='fd'):
        self.value = value

    def __call__(self):
        from gpaw.ase_interface import FDMode
        return FDMode()


class Setups(InputParameter):
    def __init__(self, value='paw'):
        self.value = value

    def __call__(self, atomic_numbers, basis, xc_name, world):
        from gpaw.setup import Setups
        return Setups(atomic_numbers,
                      self.value,
                      basis,
                      xc_name,
                      world)


class Symmetry(InputParameter):
    def __init__(self, value=None):
        if value in {None, 'off'}:
            value = {}
        self.value = value

    def __call__(self, atoms, setups, magmoms):
        from gpaw.symmetry import Symmetry as OldSymmetry
        if magmoms is None:
            ids = setups.id_a
        elif magmoms.ndim == 1:
            ids = [id + (m,) for id, m in zip(setups.id_a, magmoms)]
        else:
            ids = [id + tuple(m) for id, m in zip(setups.id_a, magmoms)]
        symmetry = OldSymmetry(ids, atoms.cell, atoms.pbc, **self.value)
        symmetry.analyze(atoms.get_scaled_positions())
        return symmetry


class Basis(InputParameter):
    def __init__(self, value=None):
        self.value = value or {}


class Magmoms(InputParameter):
    def __call__(self, atoms):
        if self.value is None:
            return atoms.get_initial_magnetic_moments()

        if isinstance(self.value, float):
            return np.zeros(len(atoms)) + self.value

        return np.array(self.value)


class KPts(InputParameter):
    def __init__(self, value=None):
        if value is None:
            value = {'size': (1, 1, 1)}
        elif not isinstance(value, dict):
            if len(value) == 3 and isinstance(value[0], int):
                value = {'size': value}
            else:
                value = {'points': np.array(value)}
        self.value = value

    def __call__(self, atoms):
        from gpaw.ase_interface import KPoints, MonkhorstPackKPoints
        if 'points' in self.value:
            return KPoints(self.values['points'])
        return MonkhorstPackKPoints(self.value['size'])


class H(InputParameter):
    pass


class GPts(InputParameter):
    pass


class InputParameters:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.params = {key: defaults[key](value)
                       for key, value in params.items()}

    def __getattr__(self, name):
        param = self.params.get(name)
        if param is not None:
            return param
        return self.defaults[name]()


@functools.lru_cache
def create_default_parameters():
    return {param.__name__.lower(): param for param in
            (cls for cls in globals().values()
             if isinstance(cls, type) and issubclass(cls, InputParameter))}
