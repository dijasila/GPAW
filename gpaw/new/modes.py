from gpaw.core import UniformGrid
from gpaw.poisson import PoissonSolver
from gpaw.fd_operators import Laplace


class FDMode:
    name = 'fd'
    stencil = 3

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

    def create_hamiltonian_operator(self, grid):
        return Hamiltonian(grid, self.stencil)


class Hamiltonian:
    def __init__(self, grid, stencil=3):
        self.gd = grid._gd
        self.kin = Laplace(self.gd, -0.5, self.stencil, grid.dtype)

    def apply(self, vt, psit, out, spin):
        self.apply(kin.apply(psit.data, out.data, psit.grid.phase_factors)
        for p, o in zip(psit.data, out.data):
            o += p * vt.data[spin]
        return out

    def create_preconditioner(self, blocksize):
        pc = PC(self.gd, self.kin, self.grid.dtype,
                self.blocksize)

        def apply(psit, residuals, out):
            ...
