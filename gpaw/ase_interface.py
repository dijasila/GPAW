from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from ase.units import Bohr

import numpy as np
from gpaw.core import UniformGrid
from gpaw.hybrids import HybridXC
from gpaw.mpi import MPIComm, Parallelization, world
from gpaw.new.input_parameters import (InputParameters,
                                       create_default_parameters)


class Logger:
    def __init__(self,
                 filename='-',
                 comm: MPIComm = None):
        comm = comm or world

        if comm.rank > 0 or filename is None:
            self.fd = open(os.devnull, 'w')
            self.close_fd = True
        elif filename == '-':
            self.fd = sys.stdout
            self.close_fd = False
        elif isinstance(filename, (str, Path)):
            self.fd = open(filename, 'w')
            self.close_fd = True
        else:
            self.fd = filename
            self.close_fd = False

        self._indent = ''

    def __del__(self):
        if self.close_fd:
            self.fd.close()

    def __call__(self, *args):
        self.fd.write(self._indent)
        print(*args, file=self.fd)

    @contextmanager
    def indent(self, text):
        self(text)
        self._indent += '  '
        yield
        self._indent = self._indent[:-2]


def GPAW(filename=None,
         *,
         txt='?',
         communicator=None,
         parallel=None,
         **parameters):

    if txt == '?' and filename:
        txt = None
    else:
        txt = '-'

    communicator = communicator or world

    log = Logger(txt, communicator)

    parallel = parallel or {}
    parallel = {**parallel, 'world': communicator}

    if filename:
        assert not parameters
        calculation = Calculation.read(filename, log, parallel)
        return calculation.ase_calculator()

    log('GPAW')
    return ASECalculator(parameters, parallel, log)


class ASECalculator:
    """This is the ASE-calculator frontend for doing a GPAW calculation."""
    def __init__(self,
                 parameters,
                 parallel,
                 log,
                 calculation=None):
        self.parameters = parameters
        self.parallel = parallel
        self.log = log
        self.calculation = calculation

    def calculate_property(self, atoms, prop: str):
        if self.calculation is None:
            self.calculation = calculate_ground_state(
                atoms, self.parameters, self.parallel, self.log)
        try:
            return self.calculation.calculate_property(atoms, prop)
        except DrasticChangesError:
            self.calculation = None
            return self.calculate_property(atoms, prop)

    def get_potential_energy(self, atoms):
        return self.calculate_property(atoms, 'energy')


def create_communicators(world: MPIComm,
                         nibzkpts: int,
                         domain: int | tuple[int, int, int] = None,
                         kpt: int = None,
                         band: int = None) -> dict[str, MPIComm]:
    parallelization = Parallelization(world, nibzkpts)
    if domain is not None:
        domain = np.prod(domain)
    parallelization.set(kpt=kpt,
                        domain=domain,
                        band=band)
    comms = parallelization.build_communicators()
    return comms


class FDMode:
    name = 'fd'

    def create_uniform_grid(self, h, gpts, cell, pbc, symmetry, comm):
        return UniformGrid(cell=cell, pbc=pbc, size=gpts, comm=comm)


class Symmetry:
    pass


class KPoints:
    def __init__(self, points):
        self.points = points


class MonkhorstPackKPoints(KPoints):
    def __init__(self, size, shift=(0, 0, 0)):
        self.size = size
        self.shift = shift
        KPoints.__init__(self, np.zeros((1, 3)))


def calculate_ground_state(atoms, parameters, parallel, log):
    default_parameters = create_default_parameters()
    params = InputParameters(parameters, default_parameters)

    world = parallel['world']

    mode = params.mode()

    xc = params.xc()

    setups = params.setups(atoms.numbers,
                           params.basis.value,
                           xc.get_setup_name(),
                           world)

    magmoms = params.magmoms(atoms)
    collinear = magmoms.ndim == 1
    if collinear:
        zmagmoms = magmoms
        magmoms = np.zeros((len(atoms), 3))
        magmoms[:, 2] = zmagmoms

    symmetry = params.symmetry(atoms, setups, magmoms)

    kpts = params.kpts(atoms)
    ibz = 'G'  # symmetry.reduce(kpts)

    d = parallel.pop('domain', None)
    k = parallel.pop('kpt', None)
    b = parallel.pop('band', None)

    if isinstance(xc, HybridXC):
        d = world.size

    communicators = create_communicators(world, len(ibz), d, k, b)

    grid = mode.create_uniform_grid(params.h.value,
                                    params.gpts.value,
                                    atoms.cell / Bohr,
                                    atoms.pbc,
                                    symmetry,
                                    comm=communicators['d'])

    if mode.name == 'fd':
        filter = create_fourier_filter(grid)
        # setups = setups.filter(filter)

    density = Density.from_superposition_of_atomic_densities(
        atoms, setups, magmoms, grid)

    if mode.name == 'pw':
        pw = mode.create_plane_waves(grid)
        layout = pw
    else:
        layout = grid

    interpolator = mode.create_interpolator(layout)
    compensation_charges = layout.atom_centered_functions()

    potential = calculate_potential(density, interpolator, compensation_charges)
    if params.random.value:
        wave_functions = WaveFunctions.from_random_numbers()

    return calculate_ground_state2()


def calculate_potential():
    density2 = density.interpolate(interpolator)
    vxc = xc(density2)

    compensation_charges.add_to(density2, density.coefs)
    vext = ...


class DrasticChangesError(Exception):
    """Atoms have changed so much that a fresh start is needed."""


def create_fourier_filter(grid):
    gamma = 1.6

    h = ((grid.icell**2).sum(1)**-0.5 / grid.size).max()

    def filter(rgd, rcut, f_r, l=0):
        gcut = np.pi / h - 2 / rcut / gamma
        ftmp = rgd.filter(f_r, rcut * gamma, gcut, l)
        f_r[:] = ftmp[:len(f_r)]

    return filter


def compare_atoms(a1, a2):
    if len(a1.numbers) != len(a2.numbers) or (a1.numbers != a2.numbers).any():
        raise DrasticChangesError
    if (a1.cell - a2.cell).abs().max() > 0.0:
        raise DrasticChangesError
    if (a1.pbc != a2.pbc).any():
        raise DrasticChangesError
    if (a1.positions - a2.positions).abs().max() > 0.0:
        return {'positions'}
    return set()


class Calculation:
    def __init__(self, atoms, parameters):
        self.atoms = atoms
        # self.parameters = parameters
        self.results = {}

    def calculate_property(self, atoms, prop):
        changes = compare_atoms(self.atoms, atoms)
        if changes:
            self.recompute_ground_state()

        if prop in self.results:
            return self.results[prop]

        if prop == 'forces':
            self.calculate_forces()
        else:
            1 / 0

        return self.results[prop]


default_parallel: dict[str, Any] = {
    'kpt': None,
    'domain': None,
    'band': None,
    'order': 'kdb',
    'stridebands': False,
    'augment_grids': False,
    'sl_auto': False,
    'sl_default': None,
    'sl_diagonalize': None,
    'sl_inverse_cholesky': None,
    'sl_lcao': None,
    'sl_lrtddft': None,
    'use_elpa': False,
    'elpasolver': '2stage',
    'buffer_size': None}
