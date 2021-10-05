from __future__ import annotations

from pathlib import Path
from typing import IO, Any
from contextlib import contextmanager
from collections import defaultdict
from time import time

from ase import Atoms
from ase.units import Bohr, Ha
from gpaw import __version__, debug
from gpaw.new.calculation import DFTCalculation
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.old import OldStuff


def GPAW(filename: str | Path | IO[str] = None,
         **kwargs) -> ASECalculator:
    """"""
    params = InputParameters(kwargs)
    txt = params.txt
    if txt == '?':
        txt = '-' if filename is None else None
    log = Logger(txt, params.parallel['world'])

    if filename is not None:
        assert len(kwargs) == 0
        ...

    write_header(log, kwargs)
    return ASECalculator(params, log)


def compare_atoms(a1, a2):
    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        return {'atomic_numbers'}
    if (a1.pbc != a2.pbc).any():
        return {'pbc'}
    if abs(a1.cell - a2.cell).max() > 0.0:
        return {'cell'}
    if abs(a1.positions - a2.positions).max() > 0.0:
        return {'positions'}
    return set()


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
        self.results = {}
        self.timer = Timer()

    def calculate_property(self, atoms: Atoms, prop: str) -> Any:
        log = self.log

        if self.calculation is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'number', 'pbc', 'cell'}:
                self.calculation = None

        if self.calculation is None:
            write_atoms(atoms, log)
            with self.timer('init'):
                self.calculation = DFTCalculation.from_parameters(
                    atoms, self.params, log)
            self.converge(atoms)
        else:
            if changes:
                write_atoms(atoms, log)
                with self.timer('move'):
                    self.calculation = self.calculation.move_atoms(atoms, log)
                self.converge(atoms)

        if prop not in self.results:
            if prop.endswith('energy'):
                free, extrapolated = self.calculation.energies(log)
                self.results['free_energy'] = free
                self.results['energy'] = extrapolated
            elif prop == 'forces':
                with self.timer('Forces'):
                    self.results['forces'] = self.calculation.forces(log)
            else:
                raise ValueError('Unknown property:', prop)

        return self.results[prop]

    def converge(self, atoms):
        with self.timer('SCF'):
            self.calculation.converge(self.log)
            self.results = {}
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

    def get_forces(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'forces') * Ha / Bohr


def write_header(log, kwargs):
    log(f' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\| - {__version__}\n')
    log(debug)
    log('Input parameters = {\n    ', end='')
    log(',\n    '.join(f'{k!r}: {v!r}' for k, v in kwargs.items()) + '}')


def write_atoms(atoms, log):
    log('\nAtoms(')
    symbols = atoms.symbols.formula.format('reduce')
    log(f'    symbols={symbols!r},')
    log('    positions=[\n       ',
        ',\n        '.join(f'({x:14.6f}, {y:14.6f}, {z:14.6f})'
                           for x, y, z in atoms.positions) +
        '],')
    log('    cell=[\n       ',
        ',\n        '.join(f'({x:14.6f}, {y:14.6f}, {z:14.6f})'
                           for x, y, z in atoms.cell) +
        '],')
    log(f'    pbc={atoms.pbc.tolist()})')


class Timer:
    def __init__(self):
        self.times = defaultdict(float)

    @contextmanager
    def __call__(self, name):
        t1 = time()
        yield
        t2 = time()
        self.times[name] += t2 - t1

    def write(self, log):
        log('\n' +
            '\n'.join(f'Time ({name + "):":12}{t:10.3f} seconds'
                      for name, t in self.times.items()))
