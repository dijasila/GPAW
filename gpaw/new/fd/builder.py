from __future__ import annotations
from gpaw.core import UniformGrid
from gpaw.new.builder import create_uniform_grid
from gpaw.new.pwfd.builder import PWFDDFTComponentsBuilder
from gpaw.new.poisson import PoissonSolverWrapper, PoissonSolver
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.fd_operators import Laplace
from gpaw.new.fd.pot_calc import UniformGridPotentialCalculator
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.new.hamiltonian import Hamiltonian


class FDDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    def __init__(self, atoms, params, nn=3, interpolation=3):
        super().__init__(atoms, params)
        self.kin_stencil_range = nn
        self.interpolation_stencil_range = interpolation

        self._nct_aR = None

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            self.mode,
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation='not fft',
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> UniformGrid:
        return self.grid.new(dtype=self.dtype)

    def get_pseudo_core_densities(self):
        if self._nct_aR is None:
            self._nct_aR = self.setups.create_pseudo_core_densities(
                self.grid, self.fracpos_ac)
        return self._nct_aR

    def create_poisson_solver(self) -> PoissonSolver:
        solver = make_poisson_solver(**self.params.poissonsolver)
        solver.set_grid_descriptor(self.fine_grid._gd)
        return PoissonSolverWrapper(solver)

    def create_potential_calculator(self):
        poisson_solver = self.create_poisson_solver()
        nct_aR = self.get_pseudo_core_densities()
        return UniformGridPotentialCalculator(self.grid,
                                              self.fine_grid,
                                              self.setups,
                                              self.xc, poisson_solver,
                                              nct_aR, self.nct_R,
                                              self.interpolation_stencil_range)

    def create_hamiltonian_operator(self, blocksize=10):
        return FDHamiltonian(self.wf_desc, self.kin_stencil_range, blocksize)

    def convert_wave_functions_from_uniform_grid(self, psit_nR):
        # No convertion needed (used for PW-mode)
        return psit_nR

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'values' not in reader.wave_functions:
            return ibzwfs

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file

        for wfs in ibzwfs:
            grid = self.wf_desc.new(kpt=wfs.kpt_c)
            index = (wfs.spin, wfs.k)
            data = reader.wave_functions.proxy('values', *index)
            data.scale = c
            wfs.psit_nX = UniformGridFunctions(grid, self.nbands,
                                               data=data)

        return ibzwfs


class FDHamiltonian(Hamiltonian):
    def __init__(self, grid, kin_stencil=3, blocksize=10):
        self.grid = grid
        self.blocksize = blocksize
        self._gd = grid._gd
        self.kin = Laplace(self._gd, -0.5, kin_stencil, grid.dtype)

    def apply(self, vt_sR, psit_nR, out, spin):
        self.kin.apply(psit_nR.data, out.data, psit_nR.desc.phase_factors_cd)
        for p, o in zip(psit_nR.data, out.data):
            o += p * vt_sR.data[spin]
        return out

    def create_preconditioner(self, blocksize):
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self._gd, self.kin, self.grid.dtype, self.blocksize)

        def apply(psit, residuals, out):
            kpt = SimpleNamespace(phase_cd=psit.desc.phase_factors_cd)
            pc(residuals.data, kpt, out=out.data)

        return apply
