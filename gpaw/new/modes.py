from __future__ import annotations

from typing import Any

import numpy as np
from ase.units import Bohr, Ha
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.fd_operators import Laplace
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.poisson import (PoissonSolver, PoissonSolverWrapper,
                              ReciprocalSpacePoissonSolver)
from gpaw.new.pot_calc import (PlaneWavePotentialCalculator,
                               UniformGridPotentialCalculator)
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.utilities.gpts import get_number_of_grid_points
import _gpaw


class Mode:
    name: str
    interpolation: str

    def __init__(self, force_complex_dtype=False):
        self.force_complex_dtype = force_complex_dtype

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
