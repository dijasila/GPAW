from gpaw.core import UniformGrid
from gpaw.poisson import PoissonSolver


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
