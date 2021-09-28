from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Sequence, TypedDict

from ase import Atoms
from gpaw import __version__
from gpaw.mpi import world, MPIComm
from gpaw.new.calculation import (Calculation, DrasticChangesError,
                                  calculate_ground_state)
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger


class ParallelKeyword(TypedDict):
    world: Sequence[int] | MPIComm


def GPAW(filename: str | Path | IO[str] = None,
         *,
         txt: str | Path | IO[str] = '?',
         parallel: ParallelKeyword = None,
         **kwargs) -> ASECalculator:
    """"""
    if txt == '?':
        txt = '-' if filename is None else None

    params = InputParameters(kwarg)

    log = Logger(txt, params.parallel['world'])

    if filename is not None:
        ...

    log(f' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\| - {__version__}\n')
    with log.indent('Input parameters ='):
        log.pp(params)

    return ASECalculator(params, log)


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 params: InputParameters,
                 log: Logger,
                 density,
                 potential,
                 ibz_wfs,
                 cfg):
        self.params = params
        self.log = log
        self.calculation = calculation

    def calculate_property(self, atoms: Atoms, prop: str) -> Any:
        if self.calculation is None:
            self.calculation = calculate_ground_state(
                atoms, self.params, self.log)
        try:
            return self.calculation.calculate_property(atoms, prop)
        except DrasticChangesError:
            self.calculation = None
            return self.calculate_property(atoms, prop)

    def get_potential_energy(self, atoms: Atoms) -> float:
        return self.calculate_property(atoms, 'energy')
