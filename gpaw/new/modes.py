from __future__ import annotations
from gpaw.core import UniformGrid
from gpaw.poisson import PoissonSolver
from gpaw.fd_operators import Laplace
from gpaw.mpi import serial_comm, MPIComm
from ase.units import Bohr, Ha
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.new.input_parameters import InputParameters


class Mode:
    def create_uniform_grid(self,
                            h: float | None,
                            gpts,
                            cell,
                            pbc,
                            symmetry,
                            comm: MPIComm = serial_comm) -> UniformGrid:
        cell = cell / Bohr
        if h is not None:
            h /= Bohr

        realspace = (self.name != 'pw' and self.interpolation != 'fft')
        if not realspace:
            pbc = (True, True, True)

        if gpts is not None:
            if h is not None:
                raise ValueError("""You can't use both "gpts" and "h"!""")
            size = gpts
        else:
            size = get_number_of_grid_points(cell, h, self, realspace,
                                             symmetry.symmetry)
        return UniformGrid(cell=cell, pbc=pbc, size=size, comm=comm)


class PWMode(Mode):
    name = 'pw'

    def __init__(self, ecut: float = 340.0):
        if ecut is not None:
            ecut /= Ha
        self.ecut = ecut

    def create_poisson_solver(self, grid, params):
        print(grid, params)

    def create_hamiltonian_operator(self, grid, blocksize=10):
        return 1 / 0


class FDMode(Mode):
    name = 'fd'
    stencil = 3
    interpolation = 'not fft'

    def create_poisson_solver(self,
                              grid: UniformGrid,
                              params: InputParameters) -> PoissonSolver:
        solver = PoissonSolver(**params)
        solver.set_grid_descriptor(grid._gd)
        return solver

    def create_hamiltonian_operator(self, grid, blocksize=10):
        return Hamiltonian(grid, self.stencil, blocksize)


class Hamiltonian:
    def __init__(self, grid, stencil=3, blocksize=10):
        self.grid = grid
        self.blocksize = blocksize
        self.gd = grid._gd
        self.kin = Laplace(self.gd, -0.5, stencil, grid.dtype)

    def apply(self, vt, psit, out, spin):
        self.kin.apply(psit.data, out.data, psit.grid.phase_factors)
        for p, o in zip(psit.data, out.data):
            o += p * vt.data[spin]
        return out

    def create_preconditioner(self, blocksize):
        from types import SimpleNamespace
        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self.gd, self.kin, self.grid.dtype, self.blocksize)

        def apply(psit, residuals, out):
            kpt = SimpleNamespace(phase_cd=psit.grid.phase_factors)
            pc(residuals.data, kpt, out=out.data)

        return apply
