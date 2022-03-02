from __future__ import annotations

from math import pi
from pathlib import Path
from typing import IO, Any, Union

from ase import Atoms
from ase.units import Bohr, Ha
from gpaw import __version__
from gpaw.new import Timer
from gpaw.new.calculation import DFTCalculation
from gpaw.new.gpw import read_gpw, write_gpw
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.typing import Array1D, Array2D


def GPAW(filename: Union[str, Path, IO[str]] = None,
         **kwargs) -> ASECalculator:
    """Create ASE-compatible GPAW calculator."""
    params = InputParameters(kwargs)
    txt = params.txt
    if txt == '?':
        txt = '-' if filename is None else None
    world = params.parallel['world']
    log = Logger(txt, world)

    if filename is not None:
        kwargs.pop('txt', None)
        assert len(kwargs) == 0
        atoms, calculation, params = read_gpw(filename, log, params.parallel)
        return ASECalculator(params, log, calculation, atoms)

    write_header(log, world, params)
    return ASECalculator(params, log)


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 params: InputParameters,
                 log: Logger,
                 calculation=None,
                 atoms=None):
        self.params = params
        self.log = log
        self.calculation = calculation

        self.atoms = atoms
        self.timer = Timer()

    def __repr__(self):
        params = []
        for key, value in self.params.items():
            val = repr(value)
            if len(val) > 40:
                val = '...'
            params.append((key, val))
        p = ', '.join(f'{key}: {val}' for key, val in params)
        return f'ASECalculator({p})'

    def calculate_property(self, atoms: Atoms, prop: str) -> Any:
        """Calculate (if not already calculated) a property.

        Must be one of

        * energy
        * forces
        * stress
        * magmom
        * magmoms
        * dipole
        """
        log = self.log

        if self.calculation is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'numbers', 'pbc', 'cell'}:
                # Start from scratch:
                if 'numbers' not in changes:
                    # Remember magmoms if there are any:
                    magmom_a = self.calculation.results.get('magmoms')
                    if magmom_a is not None:
                        atoms = atoms.copy()
                        atoms.set_initial_magnetic_moments(magmom_a)
                self.calculation = None

        if self.calculation is None:
            self.calculation = self.create_new_calculation(atoms)
            self.converge(atoms)
        elif changes:
            self.move_atoms(atoms)
            self.converge(atoms)

        if prop not in self.calculation.results:
            if prop == 'forces':
                with self.timer('Forces'):
                    self.calculation.forces(log)
            elif prop == 'stress':
                with self.timer('Stress'):
                    self.calculation.stress(log)
            elif prop == 'dipole':
                self.calculation.dipole(log)
            else:
                raise ValueError('Unknown property:', prop)

        return self.calculation.results[prop]

    def create_new_calculation(self, atoms: Atoms) -> DFTCalculation:
        with self.timer('init'):
            calculation = DFTCalculation.from_parameters(atoms, self.params,
                                                         self.log)
        return calculation

    def move_atoms(self, atoms):
        with self.timer('move'):
            self.calculation = self.calculation.move_atoms(atoms, self.log)

    def converge(self, atoms):
        """Iterate to self-consistent solution.

        Will also calculate "cheap" properties: energy, magnetic moments
        and dipole moment.
        """
        with self.timer('SCF'):
            self.calculation.converge(self.log)

        # Calculate all the cheap things:
        self.calculation.energies(self.log)
        self.calculation.dipole(self.log)
        self.calculation.magmoms(self.log)

        self.atoms = atoms.copy()
        self.calculation.write_converged(self.log)

    def __del__(self):
        self.timer.write(self.log)

    def get_potential_energy(self,
                             atoms: Atoms,
                             force_consistent: bool = False) -> float:
        return self.calculate_property(atoms,
                                       'free_energy' if force_consistent else
                                       'energy') * Ha

    def get_forces(self, atoms: Atoms) -> Array2D:
        return self.calculate_property(atoms, 'forces') * (Ha / Bohr)

    def get_stress(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'stress') * (Ha / Bohr**3)

    def get_dipole_moment(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'dipole') * Bohr

    def get_magnetic_moment(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'magmom')

    def get_magnetic_moments(self, atoms: Atoms) -> Array1D:
        return self.calculate_property(atoms, 'magmoms')

    def get_pseudo_wave_function(self, n):
        state = self.calculation.state
        return state.ibzwfs[0].wave_functions.data[n]

    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def get_fermi_level(self) -> float:
        state = self.calculation.state
        fl = state.ibzwfs.fermi_levels * Ha
        assert len(fl) == 1
        return fl[0]

    def get_homo_lumo(self, spin: int = None) -> Array1D:
        state = self.calculation.state
        return state.ibzwfs.get_homo_lumo(spin) * Ha

    def get_number_of_electrons(self):
        state = self.calculation.state
        return state.ibzwfs.nelectrons

    def get_number_of_bands(self):
        state = self.calculation.state
        return state.ibzwfs.nbands

    def get_atomic_electrostatic_potentials(self):
        _, _, Q_aL = self.calculation.pot_calc.calculate(
            self.calculation.state.density)
        Q_aL = Q_aL.gather()
        return Q_aL.data[::9] * (Ha / (4 * pi)**0.5)

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


def write_header(log, world, params):
    from gpaw.io.logger import write_header as header
    log(f' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\| - {__version__}\n')
    header(log, world)
    log('Input parameters = {\n    ', end='')
    log(',\n    '.join(f'{k!r}: {v!r}' for k, v in params.items()) + '}')


def compare_atoms(a1: Atoms, a2: Atoms) -> set[str]:
    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        return {'numbers'}
    if (a1.pbc != a2.pbc).any():
        return {'pbc'}
    if abs(a1.cell - a2.cell).max() > 0.0:
        return {'cell'}
    if abs(a1.positions - a2.positions).max() > 0.0:
        return {'positions'}
    return set()
