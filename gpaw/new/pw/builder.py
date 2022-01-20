import _gpaw
import numpy as np
from ase.units import Ha
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.new.modes import Mode
from gpaw.new.pw.poisson import ReciprocalSpacePoissonSolver
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator


class PWMode(Mode):
    name = 'pw'

    def __init__(self, ecut: float = 340.0):
        Mode.__init__(self)
        if ecut is not None:
            ecut /= Ha
        self.ecut = ecut

    def create_wf_description(self,
                              grid: UniformGrid,
                              dtype) -> PlaneWaves:

        return PlaneWaves(ecut=self.ecut, cell=grid.cell, dtype=dtype)

    def create_pseudo_core_densities(self, setups, grid, fracpos_ac):
        pw = PlaneWaves(ecut=2 * self.ecut, cell=grid.cell)
        return setups.create_pseudo_core_densities(pw, fracpos_ac)

    def create_poisson_solver(self, fine_grid_pw, params):
        return ReciprocalSpacePoissonSolver(fine_grid_pw)

    def create_potential_calculator(self,
                                    grid,
                                    fine_grid,
                                    setups,
                                    xc,
                                    poisson_solver_params,
                                    nct_ag, nct_R):
        pw = nct_ag.pw
        fine_pw = pw.new(ecut=8 * self.ecut)
        poisson_solver = self.create_poisson_solver(fine_pw,
                                                    poisson_solver_params)
        return PlaneWavePotentialCalculator(grid,
                                            fine_grid,
                                            pw,
                                            fine_pw,
                                            setups,
                                            xc,
                                            poisson_solver,
                                            nct_ag, nct_R)

    def create_hamiltonian_operator(self, grid, blocksize=10):
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


