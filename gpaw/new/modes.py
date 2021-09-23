import numpy as np
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
        gd = grid._gd
        kin = Laplace(gd, -0.5, self.stencil, grid.dtype)
        phases = {}

        def Ht(vt, psit, out, spin):
            print(vt, psit, out, spin)
            kpt = psit.grid.kpt
            ph = phases.get(tuple(kpt))
            if ph is None:
                ph = np.exp(2j * np.pi * gd.sdisp_cd * kpt[:, np.newaxis])
                phases[tuple(kpt)] = ph
            kin.apply(psit.data, out.data, ph)
            for p, o in zip(psit.data, out.data):
                o += p * vt.data[spin]
            return out

        return Ht
