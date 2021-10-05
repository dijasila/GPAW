from __future__ import annotations
import numpy as np
from gpaw.utilities import pack, unpack
from gpaw.typing import Array1D, Array3D
from gpaw.setup import Setup
from gpaw.new.xc import XCFunctional
from gpaw.core import UniformGrid, PlaneWaves
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.plane_waves import PWMapping


class Potential:
    def __init__(self,
                 vt: DistributedArrays,
                 dH: AtomArrays,
                 energies: dict[str, float]):
        self.vt = vt
        self.dv = dH
        self.energies = energies

    def dH(self, projections, out, spin=0):
        for a, I1, I2 in projections.layout.myindices:
            dh = self.dv[a][:, :, spin]
            out.data[I1:I2] = dh @ projections.data[I1:I2]
        return out


class PotentialCalculator:
    def __init__(self,
                 fine_grid: UniformGrid,
                 xc,
                 poisson_solver,
                 setups, fracpos):
        self.poisson_solver = poisson_solver
        self.xc = xc

        self.vext = fine_grid.zeros()  # initial guess for Coulomb potential
        self.total_density2 = fine_grid.empty()

        self.local_potentials = setups.create_local_potentials(fine_grid,
                                                               fracpos)
        self.v0 = self.local_potentials.evaluate()

        self.compensation_charges = setups.create_compensation_charges(
            fine_grid, fracpos)

        self.description = (poisson_solver.description +
                            str(xc) + '...')


class UniformGridPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_grid: UniformGrid,
                 fine_grid: UniformGrid,
                 setups,
                 fracpos,
                 xc,
                 poisson_solver):
        PotentialCalculator.__init__(self, fine_grid, xc, poisson_solver,
                                     setups, fracpos)

        self.interpolate = wf_grid.transformer(fine_grid)
        self.restrict = fine_grid.transformer(wf_grid)

    def calculate(self, density):
        density1 = density.density
        density2 = self.interpolate(density1, preserve_integral=True)

        grid2 = density2.grid

        vxc = grid2.zeros(density2.shape)
        e_xc = self.xc.calculate(density2, vxc)

        self.total_density2.data[:] = density2.data[:density.ndensities].sum(
            axis=0)
        e_zero = self.v0.integrate(self.total_density2)

        charge = grid2.empty()
        charge.data[:] = self.total_density2.data
        coefs = density.calculate_compensation_charge_coefficients()
        self.compensation_charges.add_to(charge, coefs)
        self.poisson_solver.solve(self.vext.data, charge.data)
        e_coulomb = 0.5 * self.vext.integrate(charge)

        potential2 = vxc
        potential2.data += self.vext.data + self.v0.data
        potential1 = self.restrict(potential2)
        e_kinetic = 0.0
        for s, (p, d) in enumerate(zip(potential1, density1)):
            e_kinetic -= p.integrate(d)
            if s < density.ndensities:
                e_kinetic += p.integrate(density.core_density)

        vnonloc, corrections = calculate_non_local_potential(
            density, self.xc,
            self.compensation_charges, self.vext)

        e_external = 0.0

        de_kinetic, de_coulomb, de_zero, de_xc, de_external = corrections
        energies = {'kinetic': e_kinetic + de_kinetic,
                    'coulomb': e_coulomb + de_coulomb,
                    'zero': e_zero + de_zero,
                    'xc': e_xc + de_xc,
                    'external': e_external + de_external}
        return Potential(potential1, vnonloc, energies)


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_pw: PlaneWaves,
                 fine_grid_pw: PlaneWaves,
                 setups,
                 fracpos,
                 xc,
                 poisson_solver):
        PotentialCalculator.__init__(self, fine_grid_pw.grid, xc,
                                     poisson_solver, setups, fracpos)

        self.pwmap = PWMapping(wf_pw, fine_grid_pw)

    def calculate(self, density):
        density1 = density.density
        rdensity1 = density1.fft(self.pwmap.pw1)
        rdensity2 = self.pwmap.pw2.zeros(density1.shape)
        rdensity2.data[:, self.pwmap.indices] = rdensity1.data
        density2 = rdensity2.ifft()

        vxc = density2.grid.zeros(density2.shape)
        e_xc = self.xc.calculate(density2, vxc)

        self.total_density2.data[:] = density2.data[:density.ndensities].sum(
            axis=0)
        e_zero = self.v0.integrate(self.total_density2)

        charge = grid2.empty()
        charge.data[:] = self.total_density2.data
        coefs = density.calculate_compensation_charge_coefficients()
        self.compensation_charges.add_to(charge, coefs)
        self.poisson_solver.solve(self.vext.data, charge.data)
        e_coulomb = 0.5 * self.vext.integrate(charge)

        potential2 = vxc
        potential2.data += self.vext.data + self.v0.data
        potential1 = self.restrict(potential2)
        e_kinetic = 0.0
        for s, (p, d) in enumerate(zip(potential1, density1)):
            e_kinetic -= p.integrate(d)
            if s < density.ndensities:
                e_kinetic += p.integrate(density.core_density)

        vnonloc, corrections = calculate_non_local_potential(
            density, self.xc,
            self.compensation_charges, self.vext)

        e_external = 0.0

        de_kinetic, de_coulomb, de_zero, de_xc, de_external = corrections
        energies = {'kinetic': e_kinetic + de_kinetic,
                    'coulomb': e_coulomb + de_coulomb,
                    'zero': e_zero + de_zero,
                    'xc': e_xc + de_xc,
                    'external': e_external + de_external}
        return Potential(potential1, vnonloc, energies)


def calculate_non_local_potential(density, xc,
                                  compensation_charges, vext):
    coefs = compensation_charges.integrate(vext)
    vnonloc = density.density_matrices.new()
    energy_corrections = np.zeros(5)
    for a, D in density.density_matrices.items():
        Q = coefs[a]
        setup = density.setups[a]
        dH, energies = calculate_non_local_potential1(setup, xc, D, Q)
        vnonloc[a][:] = dH
        energy_corrections += energies

    return vnonloc, energy_corrections


def calculate_non_local_potential1(setup: Setup,
                                   xc: XCFunctional,
                                   D: Array3D,
                                   Q: Array1D) -> tuple[Array3D, Array1D]:
    ndensities = 2 if D.shape[2] == 2 else 1
    d = np.array([pack(D1) for D1 in D.T])

    d1 = d[:ndensities].sum(0)

    h1 = (setup.K_p + setup.M_p +
          setup.MB_p + 2.0 * setup.M_pp @ d1 +
          setup.Delta_pL @ Q)
    e_kinetic = setup.K_p @ d1 + setup.Kc
    e_zero = setup.MB + setup.MB_p @ d1
    e_coulomb = setup.M + d1 @ (setup.M_p + setup.M_pp @ d1)

    h = np.zeros_like(d)
    h[:ndensities] = h1
    e_xc = xc.calculate_paw_correction(setup, d, h)
    e_kinetic -= (d * h).sum().real

    e_external = 0.0

    H = unpack(h).T

    return H, np.array([e_kinetic, e_coulomb, e_zero, e_xc, e_external])
