from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Union

from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import __version__
from gpaw.new import Timer
from gpaw.new.calculation import DFTCalculation
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.old import OldStuff, read_gpw
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
        calculation, params = read_gpw(filename, log, params.parallel)
        return ASECalculator(params, log, calculation)

    write_header(log, world, kwargs)
    return ASECalculator(params, log)


class ASECalculator(OldStuff):
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 params: InputParameters,
                 log: Logger,
                 calculation=None):
        self.params = params
        self.log = log
        self.calculation = calculation

        self.atoms = None
        self.timer = Timer()

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
                if 'numbers' not in changes:
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
        self.calculation.energies(self.log)
        # self.calculation.dipole(self.log)
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


def write_header(log, world, kwargs):
    from gpaw.io.logger import write_header as header
    log(f' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\| - {__version__}\n')
    header(log, world)
    log('Input parameters = {\n    ', end='')
    log(',\n    '.join(f'{k!r}: {v!r}' for k, v in kwargs.items()) + '}')


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
