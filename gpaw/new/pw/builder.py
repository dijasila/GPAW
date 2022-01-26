import _gpaw
import numpy as np
from ase.units import Ha
from gpaw.core import PlaneWaves
from gpaw.new.pw.poisson import ReciprocalSpacePoissonSolver
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from gpaw.new.builder import DFTComponentsBuilder, create_uniform_grid


class PWDFTComponentsBuilder(DFTComponentsBuilder):
    interpolation = 'fft'

    def __init__(self, atoms, params):
        self.ecut = params.mode.get('ecut', 340) / Ha
        super().__init__(atoms, params)

        self._nct_ag = None

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            'pw',
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation='fft',
            ecut=self.ecut,
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> PlaneWaves:
        assert self.grid.comm.size == 1
        return PlaneWaves(ecut=self.ecut,
                          cell=self.grid.cell,
                          dtype=self.dtype)

    def get_pseudo_core_densities(self):
        if self._nct_ag is None:
            pw = PlaneWaves(ecut=2 * self.ecut, cell=self.grid.cell)
            self._nct_ag = self.setups.create_pseudo_core_densities(
                pw, self.fracpos_ac)
        return self._nct_ag

    def create_poisson_solver(self, fine_grid_pw, params):
        return ReciprocalSpacePoissonSolver(fine_grid_pw)

    def create_potential_calculator(self):
        nct_ag = self.get_pseudo_core_densities()
        pw = nct_ag.pw
        fine_pw = pw.new(ecut=8 * self.ecut)
        poisson_solver = self.create_poisson_solver(
            fine_pw,
            self.params.poissonsolver)
        return PlaneWavePotentialCalculator(self.grid,
                                            self.fine_grid,
                                            pw,
                                            fine_pw,
                                            self.setups,
                                            self.xc,
                                            poisson_solver,
                                            nct_ag, self.nct_R)

    def create_hamiltonian_operator(self, blocksize=10):
        return PWHamiltonian()


class PWHamiltonian:
    def apply(self, vt_sR, psit_nG, out, spin):
        out_nG = out
        vt_R = vt_sR.data[spin]
        np.multiply(psit_nG.desc.ekin_G, psit_nG.data, out_nG.data)
        f_R = None
        for p_G, o_G in zip(psit_nG, out_nG):
            f_R = p_G.ifft(grid=vt_sR.desc, out=f_R)
            f_R.data *= vt_R
            o_G.data += f_R.fft(pw=p_G.desc).data
        return out_nG

    def create_preconditioner(self, blocksize):
        return precondition


def precondition(psit, residuals, out):
    G2 = psit.desc.ekin_G * 2
    for r, o, ekin in zip(residuals.data, out.data, psit.norm2('kinetic')):
        _gpaw.pw_precond(G2, r, ekin, o)
