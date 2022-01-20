from __future__ import annotations
from gpaw.core import UniformGrid
from gpaw.new.builder import DFTComponentsBuilder, create_uniform_grid
from gpaw.new.poisson import PoissonSolverWrapper, PoissonSolver
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.fd_operators import Laplace
from gpaw.new.fd.pot_calc import UniformGridPotentialCalculator


class FDDFTComponentsBuilder(DFTComponentsBuilder):
    stencil = 3
    interpolation = 'not fft'

    def __init__(self, atoms, params):
        super().__init__(atoms, params)

        self.grid = create_uniform_grid(
            'fd',
            params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=params.h,
            interpolation='not fft',
            comm=self.communicators['d'])

        self.fine_grid = self.grid.new(size=self.grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]

    def create_wf_description(self) -> UniformGrid:
        return self.grid.new(dtype=self.dtype)

    def create_pseudo_core_densities(self):
        return self.setups.create_pseudo_core_densities(self.grid,
                                                        self.fracpos_ac)

    def create_poisson_solver(self) -> PoissonSolver:
        solver = make_poisson_solver(**self.params.poissonsolver)
        solver.set_grid_descriptor(self.fine_grid._gd)
        return PoissonSolverWrapper(solver)

    def create_potential_calculator(self):
        poisson_solver = self.create_poisson_solver()
        return UniformGridPotentialCalculator(self.grid,
                                              self.fine_grid,
                                              self.setups,
                                              self.xc, poisson_solver,
                                              self.nct_aX, self.nct_R)

    def create_hamiltonian_operator(self, blocksize=10):
        return FDHamiltonian(self.wf_desc, self.stencil, blocksize)


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
