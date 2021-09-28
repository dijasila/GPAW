from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Sequence, TypedDict

from ase import Atoms
from gpaw import __version__
from gpaw.mpi import MPIComm
from gpaw.new.calculation import DFTCalculation
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger


class ParallelKeyword(TypedDict):
    world: Sequence[int] | MPIComm


def GPAW(filename: str | Path | IO[str] = None,
         *,
         txt: str | Path | IO[str] = '?',
         **kwargs) -> ASECalculator:
    """"""
    if txt == '?':
        txt = '-' if filename is None else None

    params = InputParameters(kwargs)

    log = Logger(txt, params.parallel['world'])

    if filename is not None:
        assert len(kwargs) == 0
        ...

    log(f' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\| - {__version__}\n')
    with log.indent('Input parameters ='):
        log.pp(params)

    return ASECalculator(params, log)


def compare_atoms(a1, a2):
    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        return {'atomic_numbers'}
    if (a1.pbc != a2.pbc).any():
        return {'pbc'}
    if (a1.cell - a2.cell).abs().max() > 0.0:
        return {'cell'}
    if (a1.positions - a2.positions).abs().max() > 0.0:
        return {'positions'}
    return set()


def write_info(atoms, log):
    with log.indent('\nAtoms('):
        symbols = atoms.symbols.formula.format('reduce')
        log(f'symbols = {symbols!r},')
        with log.indent('positions ='):
            log.pp(atoms.positions.tolist())
        with log.indent('cell ='):
            log.pp(atoms.cell.tolist())
        with log.indent('pbc ='):
            log.pp(atoms.pbc.tolist())
        log(')')


class ASECalculator:
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
        if self.calculation is None:
            self.calculation = DFTCalculation.from_parameters(
                atoms, self.params, self.log)
            self.calculation.converge(self.log)
            self.results = {}
        else:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'number', 'pbc', 'cell'}:
                self.calculation = None
                return self.calculate_property(atoms, prop)
            if changes:
                self.calculation.move(atoms.get_scaled_positions(), self.log)
                self.calculation.converge(self.log)
                self.results = {}

        if prop not in self.results:
            if prop == 'forces':
                self.results[prop] = self.calculation.forces()
            elif prop == 'energy':
                self.results[prop] = self.calculation.energy()
            else:
                1 / 0

        return self.results[prop]

    def get_potential_energy(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'energy')
