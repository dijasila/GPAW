from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Union

import numpy as np
from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import __version__, debug
from gpaw.new import Timer
from gpaw.new.calculation import DFTCalculation
from gpaw.new.input_parameters import InputParameters
from gpaw.new.logger import Logger
from gpaw.new.old import OldStuff, read_gpw


def GPAW(filename: Union[str, Path, IO[str]] = None,
         **kwargs) -> ASECalculator:
    """"""
    params = InputParameters(kwargs)
    txt = params.txt
    if txt == '?':
        txt = '-' if filename is None else None
    log = Logger(txt, params.parallel['world'])

    if filename is not None:
        kwargs.pop('txt')
        assert len(kwargs) == 0
        calculation = read_gpw(filename, log)
        return calculation.ase_interface()

    write_header(log, kwargs)
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
        log = self.log

        if self.calculation is not None:
            changes = compare_atoms(self.atoms, atoms)
            if changes & {'number', 'pbc', 'cell'}:
                self.calculation = None

        if self.calculation is None:
            self.calculation = self.create_new_calculation(atoms)
            self.converge(atoms)
        elif changes:
            self.move_atoms(atoms)
            self.converge(atoms)

        if prop not in self.calculation.results:
            if prop.endswith('energy'):
                self.calculation.energies(log)
            elif prop == 'forces':
                with self.timer('Forces'):
                    self.calculation.forces(log)
            else:
                raise ValueError('Unknown property:', prop)

        return self.calculation.results[prop]

    def create_new_calculation(self, atoms: Atoms) -> DFTCalculation:
        with self.timer('init'):
            calculation = DFTCalculation.from_parameters(atoms, self.params,
                                                         self.log)
        write_atoms(calculation.cfg, self.log)
        return calculation

    def move_atoms(self, atoms):
        write_atoms(atoms, self.log)
        with self.timer('move'):
            self.calculation = self.calculation.move_atoms(atoms, self.log)

    def converge(self, atoms):
        with self.timer('SCF'):
            self.calculation.converge(self.log)
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


def write_atoms(cfg, log):
    from gpaw.output import print_cell, print_positions
    magmoms = cfg.initial_magmoms
    if magmoms is None:
        magmoms = np.zeros((len(cfg.atoms), 3))
    print_positions(cfg.atoms, log, magmoms)
    print_cell(cfg.grid._gd, cfg.atoms.pbc, log)


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
