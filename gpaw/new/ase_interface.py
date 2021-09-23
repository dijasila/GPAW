from __future__ import annotations

from pathlib import Path
from typing import IO, Any

from ase import Atoms
from gpaw.mpi import world
from gpaw.new.calculation import (Calculation, DrasticChangesError,
                                  calculate_ground_state)
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger


def GPAW(filename: str | Path | IO[str] = None,
         *,
         txt: str | Path | IO[str] | None = '?',
         **parameters) -> ASECalculator:

    if txt == '?' and filename:
        txt = None
    else:
        txt = '-'

    params = InputParameters(parameters)
    comm = params.parallel['world'] or world
    log = Logger(txt, comm)

    if filename:
        parameters.pop('parallel')
        assert not parameters
        calculation = Calculation.read(filename, log, comm)
        return calculation.ase_calculator()

    log(' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\|\n')
    with log.indent('Input parameters:'):
        log.pp(params.params)
    return ASECalculator(params, log)


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 params: InputParameters,
                 log: Logger,
                 calculation: Calculation = None):
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
