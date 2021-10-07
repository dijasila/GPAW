from __future__ import annotations

from typing import Any

import numpy as np
from ase.units import Bohr, Ha
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.fd_operators import Laplace
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.poisson import (PoissonSolver, PoissonSolverWrapper,
                              ReciprocalSpacePoissonSolver)
from gpaw.new.potential import (PlaneWavePotentialCalculator,
                                UniformGridPotentialCalculator)
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.utilities.gpts import get_number_of_grid_points


class Mode:
    name: str
    interpolation: str

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
        if ecut is not None:
            ecut /= Ha
        self.ecut = ecut

    def create_poisson_solver(self, fine_grid_pw, params):
        return ReciprocalSpacePoissonSolver(fine_grid_pw)

    def create_potential_calculator(self,
                                    wf_pw: PlaneWaves,
                                    fine_grid, setups,
                                    fracpos,
                                    xc,
                                    poisson_solver_params):
        fine_grid_pw = PlaneWaves(ecut=4 * wf_pw.ecut, grid=fine_grid)
        poisson_solver = self.create_poisson_solver(fine_grid_pw,
                                                    poisson_solver_params)
        return PlaneWavePotentialCalculator(wf_pw, fine_grid_pw,
                                            setups, fracpos,
                                            xc, poisson_solver)

    def create_hamiltonian_operator(self, grid, blocksize=10):
        return PWHamiltonian()


class PWHamiltonian:
    def __init__(self):
        ...

    def apply(self, vt, psit, out, spin):
        v = vt.data[spin]
        np.multiply(psit.pw.ekin, psit.data, out.data)
        for p, o in zip(psit, out):
            o.data += (p.ifft() * v).fft()
        return out

    def create_preconditioner(self, blocksize):
        1 / 0
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self.gd, self.kin, self.grid.dtype, self.blocksize)

        def apply(psit, residuals, out):
            kpt = SimpleNamespace(phase_cd=psit.grid.phase_factors)
            pc(residuals.data, kpt, out=out.data)

        return apply


class FDMode(Mode):
    name = 'fd'
    stencil = 3
    interpolation = 'not fft'

    def create_poisson_solver(self,
                              grid: UniformGrid,
                              params: dict[str, Any]) -> PoissonSolver:
        solver = make_poisson_solver(**params)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)

    def create_potential_calculator(self, wf_grid, fine_grid, setups,
                                    fracpos, xc, poisson_solver_params):
        poisson_solver = self.create_poisson_solver(fine_grid,
                                                    poisson_solver_params)
        return UniformGridPotentialCalculator(wf_grid, fine_grid,
                                              setups, fracpos,
                                              xc, poisson_solver)

    def create_hamiltonian_operator(self, grid, blocksize=10):
        return FDHamiltonian(grid, self.stencil, blocksize)


class FDHamiltonian:
    def __init__(self, grid, stencil=3, blocksize=10):
        self.grid = grid
        self.blocksize = blocksize
        self.gd = grid._gd
        self.kin = Laplace(self.gd, -0.5, stencil, grid.dtype)

    def apply(self, vt, psit, out, spin):
        self.kin.apply(psit.data, out.data, psit.grid.phase_factors)
        for p, o in zip(psit.data, out.data):
            o += p * vt.data[spin]
        return out

    def create_preconditioner(self, blocksize):
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self.gd, self.kin, self.grid.dtype, self.blocksize)

        def apply(psit, residuals, out):
            kpt = SimpleNamespace(phase_cd=psit.grid.phase_factors)
            pc(residuals.data, kpt, out=out.data)

        return apply
