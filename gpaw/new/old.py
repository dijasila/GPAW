from __future__ import annotations
import numpy as np
from ase.units import Ha
from gpaw.typing import Array1D
from gpaw.new.gpw import write_gpw

methods = []


def add_method(func):
    methods.append((func.__name__, func))
    return func


@add_method
def get_pseudo_wave_function(self, n):
    state = self.calculation.state
    return state.ibzwfs[0].wave_functions.data[n]


@add_method
def get_atoms(self):
    atoms = self.atoms.copy()
    atoms.calc = self
    return atoms


@add_method
def get_fermi_level(self) -> float:
    state = self.calculation.state
    fl = state.ibzwfs.fermi_levels * Ha
    assert len(fl) == 1
    return fl[0]


@add_method
def get_homo_lumo(self, spin: int = None) -> Array1D:
    state = self.calculation.state
    return state.ibzwfs.get_homo_lumo(spin) * Ha


@add_method
def get_number_of_electrons(self):
    state = self.calculation.state
    return state.ibzwfs.nelectrons


@add_method
def get_atomic_electrostatic_potentials(self):
    _, _, Q_aL = self.calculation.pot_calc.calculate(
        self.calculation.state.density)
    Q_aL = Q_aL.gather()
    return Q_aL.data[::9] * (Ha / (4 * np.pi)**0.5)


@add_method
def write(self, filename, mode=''):
    """Write calculator object to a file.

    Parameters
    ----------
    filename
        File to be written
    mode
        Write mode. Use ``mode='all'``
        to include wave functions in the file.
    """
    self.log(f'Writing to {filename} (mode={mode!r})\n')

    write_gpw(filename, self.atoms, self.params,
              self.calculation, skip_wfs=mode != 'all')
