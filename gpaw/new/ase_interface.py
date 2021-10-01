from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Sequence, TypedDict
from contextlib import contextmanager
from collections import defaultdict
from time import time

from ase import Atoms
from gpaw import __version__, debug
from gpaw.mpi import MPIComm
from gpaw.new.calculation import DFTCalculation
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.old import OldStuff


class ParallelKeyword(TypedDict):
    world: Sequence[int] | MPIComm


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

    def calculate_property(self, atoms: Atoms, prop: str) -> Any:
        log = self.log
        timer = Timer()
        if self.calculation is None:
            write_atoms(atoms, log)
            with timer('init'):
                self.calculation = DFTCalculation.from_parameters(
                    atoms, self.params, log)
            with timer('SCF'):
                self.calculation.converge(log)
            self.results = {}
            self.atoms = atoms.copy()
            changes = {'number', 'pbc', 'cell'}
        else:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'number', 'pbc', 'cell'}:
                self.calculation = None
                return self.calculate_property(atoms, prop)
            if changes:
                fracpos = atoms.get_scaled_positions()
                with timer('move'):
                    self.calculation.move(fracpos, log)
                with timer('SCF'):
                    self.calculation.converge(log)
                self.results = {}

        if changes:
            self.calculation.write_converged(log)

        if prop not in self.results:
            if prop == 'energy':
                self.results[prop] = self.calculation.energy(log)
            elif prop == 'forces':
                with timer('Forces'):
                    self.results[prop] = self.calculation.forces(log)
            else:
                raise ValueError('Unknown property:', prop)

        timer.write(log)

        return self.results[prop]

    def get_potential_energy(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'energy')

    def get_forces(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'forces')


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
        log('\nTiming (seconds):\n   ',
            '\n    '.join(f'{name + ":":10}{t:10.3f}'
                          for name, t in self.times.items()))
