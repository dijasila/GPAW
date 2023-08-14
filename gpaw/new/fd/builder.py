from __future__ import annotations

from gpaw.core import UniformGrid
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.fd_operators import Laplace
from gpaw.new import zips
from gpaw.new.builder import create_uniform_grid
from gpaw.new.fd.pot_calc import UniformGridPotentialCalculator
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.new.pwfd.builder import PWFDDFTComponentsBuilder
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.fd_operators import Gradient


class FDDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    def __init__(self, atoms, params, *, comm, nn=3, interpolation=3):
        super().__init__(atoms,
                         params,
                         comm=comm)
        assert not self.soc
        self.kin_stencil_range = nn
        self.interpolation_stencil_range = interpolation

        self._nct_aR = None
        self._tauct_aR = None

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
                self.grid, self.fracpos_ac, atomdist=self.atomdist, xp=self.xp)
        return self._nct_aR

    def get_pseudo_core_ked(self):
        if self._tauct_aR is None:
            self._tauct_aR = self.setups.create_pseudo_core_ked(
                self.grid, self.fracpos_ac, atomdist=self.atomdist)
        return self._tauct_aR

    def create_poisson_solver(self) -> PoissonSolver:
        solver = make_poisson_solver(**self.params.poissonsolver)
        solver.set_grid_descriptor(self.fine_grid._gd)
        return PoissonSolverWrapper(solver)

    def create_potential_calculator(self):
        poisson_solver = self.create_poisson_solver()
        return UniformGridPotentialCalculator(
            self.grid, self.fine_grid, self.setups, self.xc, poisson_solver,
            fracpos_ac=self.fracpos_ac, atomdist=self.atomdist,
            interpolation_stencil_range=self.interpolation_stencil_range,
            xp=self.xp)

    def create_hamiltonian_operator(self, blocksize=10):
        return FDHamiltonian(self.wf_desc, self.kin_stencil_range, blocksize)

    def convert_wave_functions_from_uniform_grid(self,
                                                 C_nM,
                                                 basis_set,
                                                 kpt_c,
                                                 q):
        grid = self.grid.new(kpt=kpt_c, dtype=self.dtype)
        psit_nR = grid.zeros(self.nbands, self.communicators['b'])
        mynbands = len(C_nM.data)
        basis_set.lcao_to_grid(C_nM.data, psit_nR.data[:mynbands], q)
        return psit_nR

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'coefficients' in reader.wave_functions:
            name = 'coefficients'
        elif 'values' in reader.wave_functions:
            name = 'values'
        else:
            return ibzwfs

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file

        for wfs in ibzwfs:
            grid = self.wf_desc.new(kpt=wfs.kpt_c)
            index = (wfs.spin, wfs.k)
            data = reader.wave_functions.proxy(name, *index)
            data.scale = c
            if self.communicators['w'].size == 1:
                wfs.psit_nX = UniformGridFunctions(grid, self.nbands,
                                                   data=data)
            else:
                band_comm = self.communicators['b']
                wfs.psit_nX = UniformGridFunctions(
                    grid, self.nbands,
                    comm=band_comm)
                if grid.comm.rank == 0:
                    mynbands = (self.nbands +
                                band_comm.size - 1) // band_comm.size
                    n1 = min(band_comm.rank * mynbands, self.nbands)
                    n2 = min((band_comm.rank + 1) * mynbands, self.nbands)
                    assert wfs.psit_nX.mydims[0] == n2 - n1
                    data = data[n1:n2]  # read from file
                wfs.psit_nX.scatter_from(data)

        return ibzwfs


class FDHamiltonian(Hamiltonian):
    def __init__(self, grid, kin_stencil=3, blocksize=10):
        self.grid = grid
        self.blocksize = blocksize
        self._gd = grid._gd
        self.kin = Laplace(self._gd, -0.5, kin_stencil, grid.dtype)

        # For MGGA:
        self.grad_v = []

    def apply(self,
              vt_sR: UniformGridFunctions,
              dedtaut_sR: UniformGridFunctions | None,
              psit_nR: UniformGridFunctions,
              out: UniformGridFunctions,
              spin: int) -> UniformGridFunctions:
        self.apply_local_potential(vt_sR[spin], psit_nR, out)
        if dedtaut_sR is not None:
            self.apply_mgga(dedtaut_sR[spin], psit_nR, out)
        return out

    def apply_local_potential(self,
                              vt_R: UniformGridFunctions,
                              psit_nR: UniformGridFunctions,
                              out: UniformGridFunctions,
                              ) -> None:
        self.kin(psit_nR, out)
        for p, o in zips(psit_nR.data, out.data):
            o += p * vt_R.data

    def apply_mgga(self,
                   dedtaut_R: UniformGridFunctions,
                   psit_nR: UniformGridFunctions,
                   vt_nR: UniformGridFunctions) -> None:
        if len(self.grad_v) == 0:
            grid = psit_nR.desc
            self.grad_v = [
                Gradient(grid._gd, v, n=3, dtype=grid.dtype)
                for v in range(3)]

        tmp_R = psit_nR.desc.empty()
        for psit_R, out_R in zips(psit_nR, vt_nR):
            for grad in self.grad_v:
                grad(psit_R, tmp_R)
                grad(dedtaut_R * tmp_R, tmp_R)
                tmp_R.data *= 0.5
                out_R.data -= tmp_R.data

    def create_preconditioner(self, blocksize):
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self._gd, self.kin, self.grid.dtype, self.blocksize)

        def apply(psit, residuals, out):
            kpt = SimpleNamespace(phase_cd=psit.desc.phase_factor_cd)
            pc(residuals.data, kpt, out=out.data)

        return apply
