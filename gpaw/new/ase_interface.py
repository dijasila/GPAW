from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, IO, Iterator
from ase.units import Bohr
from ase import Atoms

import numpy as np
from gpaw.core import UniformGrid
from gpaw.hybrids import HybridXC
from gpaw.mpi import MPIComm, Parallelization, world
from gpaw.new.input_parameters import InputParameters
from gpaw.new.density import Density
from gpaw.poisson import PoissonSolver
from gpaw.new.wfs import IBZWaveFunctions


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

    def __del__(self) -> None:
        if self.close_fd:
            self.fd.close()

    def __call__(self, *args) -> None:
        self.fd.write(self._indent)
        print(*args, file=self.fd)

    def comment(self, text):
        print('# ' + '\n# '.join(text.splitlines()), file=self.fd)

    @contextmanager
    def indent(self, text: str) -> Iterator:
        self(text)
        self._indent += '  '
        yield
        self._indent = self._indent[:-2]


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

    log.comment(' __  _  _\n| _ |_)|_||  |\n|__||  | ||/\\|\n')

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

    def create_uniform_grid(self,
                            h,
                            gpts,
                            cell,
                            pbc,
                            symmetry,
                            comm) -> UniformGrid:
        return UniformGrid(cell=cell, pbc=pbc, size=gpts, comm=comm)

    def create_poisson_solver(self, grid, params):
        solver = PoissonSolver(**params)
        solver.set_grid_descriptor(grid._gd)
        return solver


class XCFunctional:
    def __init__(self, xc):
        self.xc = xc
        self.setup_name = xc.get_setup_name()

    def calculate(self, density, out) -> float:
        return self.xc.calculate(density.grid._gd, density.data, out.data)

    def calculate_paw_correction(self, setup, d, h):
        return self.xc.calculate_paw_correction(setup, d, h)


class Symmetry:
    def __init__(self, symmetry):
        self.symmetry = symmetry

    def reduce(self, bz):
        return IBZ(self, bz, [0], [1.0])


class BZ:
    def __init__(self, points):
        self.points = points


class MonkhorstPackKPoints(BZ):
    def __init__(self, size, shift=(0, 0, 0)):
        self.size = size
        self.shift = shift
        BZ.__init__(self, np.zeros((1, 3)))


class IBZ:
    def __init__(self, symmetry, bz, indices, weights):
        self.symmetry = symmetry
        self.bz = bz
        self.weights = weights
        self.points = bz.points[indices]

    def __len__(self):
        return len(self.points)

    def ranks(self, comm):
        return [0]


def calculate_ground_state(atoms, params, log):
    from gpaw.new.hamiltonian import Hamiltonian
    mode = params.mode
    base = Base.from_parameters(atoms, params)
    setups = base.setups

    density = Density.from_superposition(base, params.charge, params.hund)

    poisson_solver = mode.create_poisson_solver(base.grid2,
                                                params.poissonsolver)
    if mode.name == 'pw':
        pw = mode.create_plane_waves(base.grid)
        basis = pw
    else:
        basis = base.grid

    hamiltonian = Hamiltonian(basis, base, poisson_solver)

    potential1, vnonloc, energies = hamiltonian.calculate_potential(density)

    nbands = params.nbands(setups, density.charge, base.magmoms,
                           mode.name == 'lcao')

    if params.random:
        ibzwfs = IBZWaveFunctions.from_random_numbers(base, nbands)

    scf = SCFLoop(ibzwfs, density, hamiltonian, ..., ...)
    for _ in scf():
        ...

    return


class SCFLoop:
    def __init__(self, ibzwfs, density, hamiltonian, eigensolver, mixer):
        ibzwfs.mykpts[0].orthonormalize()
        ...


class Base:
    def __init__(self,
                 positions,
                 setups,
                 communicators,
                 grid,
                 xc,
                 ibz,
                 magmoms=None):
        self.positions = positions
        self.setups = setups
        self.magmoms = magmoms
        self.communicators = communicators
        self.ibz = ibz
        self.grid = grid
        self.xc = xc

        self.grid2 = grid.new(size=grid.size * 2)
        # decomposition=[2 * d for d in grid.decomposition]

    @classmethod
    def from_parameters(self, atoms, params):
        parallel = params.parallel
        world = parallel['world']
        mode = params.mode
        xc = params.xc

        setups = params.setups(atoms.numbers,
                               params.basis,
                               xc.setup_name,
                               world)

        magmoms = params.magmoms(atoms)

        symmetry = params.symmetry(atoms, setups, magmoms)

        bz = params.kpts(atoms)
        ibz = symmetry.reduce(bz)

        d = parallel.pop('domain', None)
        k = parallel.pop('kpt', None)
        b = parallel.pop('band', None)

        if isinstance(xc, HybridXC):
            d = world.size

        communicators = create_communicators(world, len(ibz), d, k, b)

        grid = mode.create_uniform_grid(params.h,
                                        params.gpts,
                                        atoms.cell / Bohr,
                                        atoms.pbc,
                                        symmetry,
                                        comm=communicators['d'])

        if mode.name == 'fd':
            pass  # filter = create_fourier_filter(grid)
            # setups = setups.filter(filter)

        return Base(atoms.get_scaled_positions(),
                    setups, communicators, grid, xc, ibz, magmoms)


def calculate_ground_state2(wave_functions, potential):
    ...


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

    @classmethod
    def read(self, filename, log, parallel):
        ...

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
