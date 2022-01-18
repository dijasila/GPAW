from __future__ import annotations
from gpaw.core import UniformGrid
from gpaw.new.modes import Mode
from gpaw.new.poisson import PoissonSolverWrapper, PoissonSolver
from typing import Any
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.fd_operators import Laplace
from gpaw.new.fd.pot_calc import UniformGridPotentialCalculator


class FDMode(Mode):
    name = 'fd'
    stencil = 3
    interpolation = 'not fft'

    def create_wf_description(self,
                              grid: UniformGrid,
                              dtype) -> UniformGrid:
        return grid.new(dtype=dtype)

    def create_pseudo_core_densities(self, setups, grid, fracpos_ac):
        return setups.create_pseudo_core_densities(grid, fracpos_ac)

    def create_poisson_solver(self,
                              grid: UniformGrid,
                              params: dict[str, Any]) -> PoissonSolver:
        solver = make_poisson_solver(**params)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)

    def create_potential_calculator(self, grid, fine_grid,
                                    setups,
                                    xc,
                                    poisson_solver_params,
                                    nct_aR, nct_R):
        poisson_solver = self.create_poisson_solver(fine_grid,
                                                    poisson_solver_params)
        return UniformGridPotentialCalculator(grid,
                                              fine_grid,
                                              setups,
                                              xc, poisson_solver,
                                              nct_aR, nct_R)

    def create_hamiltonian_operator(self, grid, blocksize=10):
        return FDHamiltonian(grid, self.stencil, blocksize)


class FDHamiltonian:
    def __init__(self, grid, stencil=3, blocksize=10):
        self.grid = grid
        self.blocksize = blocksize
        self.gd = grid._gd
        self.kin = Laplace(self.gd, -0.5, stencil, grid.dtype)

    def apply(self, vt_sR, psit_nR, out, spin):
        self.kin.apply(psit_nR.data, out.data, psit_nR.desc.phase_factors_cd)
        for p, o in zip(psit_nR.data, out.data):
            o += p * vt_sR.data[spin]
        return out

    def create_preconditioner(self, blocksize):
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self.gd, self.kin, self.grid.dtype, self.blocksize)

        def apply(psit, residuals, out):
            kpt = SimpleNamespace(phase_cd=psit.desc.phase_factors_cd)
            pc(residuals.data, kpt, out=out.data)

        return apply
