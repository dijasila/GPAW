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
                 xc,
                 poisson_solver):
        self.poisson_solver = poisson_solver
        self.xc = xc
        self.vbar = self.vbar_acf.evaluate()

    def __str__(self):
        return f'\n{self.poisson_solver}\n{self.xc}'

    def calculate(self, density):
        energies, potential= self._calculate(density)

        vnonloc, corrections = calculate_non_local_potential(
            density, self.xc, self.ghat_acf, self.vHt)

        de_kinetic, de_coulomb, de_zero, de_xc, de_external = corrections
        energies['kinetic'] += de_kinetic
        energies['coulomb'] += de_coulomb
        energies['zero'] += de_zero
        energies['xc'] += de_xc
        energies['external'] += de_external

        return Potential(potential, vnonloc, energies)


class UniformGridPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_grid: UniformGrid,
                 fine_grid: UniformGrid,
                 setups,
                 fracpos,
                 xc,
                 poisson_solver):
        self.vHt = fine_grid.zeros()  # initial guess for Coulomb potential
        self.nt = fine_grid.empty()
        self.vt = wf_grid.empty()

        self.vbar_acf = setups.create_local_potentials(fine_grid, fracpos)
        self.ghat_acf = setups.create_compensation_charges(fine_grid, fracpos)

        self.interpolate = wf_grid.transformer(fine_grid)
        self.restrict = fine_grid.transformer(wf_grid)

        PotentialCalculator.__init__(self, xc, poisson_solver)

    def _calculate(self, density):
        density1 = density.density
        density2 = self.interpolate(density1, preserve_integral=True)

        grid2 = density2.grid

        vxct = grid2.zeros(density2.shape)
        e_xc = self.xc.calculate(density2, vxct)

        self.nt.data[:] = density2.data[:density.ndensities].sum(axis=0)
        e_zero = self.vbar.integrate(self.nt)

        charge = grid2.empty()
        charge.data[:] = self.nt.data
        coefs = density.calculate_compensation_charge_coefficients()
        self.ghat_acf.add_to(charge, coefs)
        self.poisson_solver.solve(self.vHt, charge)
        e_coulomb = 0.5 * self.vHt.integrate(charge)

        potential2 = vxct
        potential2.data += self.vHt.data + self.vbar.data
        potential1 = self.restrict(potential2)
        e_kinetic = 0.0
        self.vt.data[:] = 0.0
        for spin, (vt, nt) in enumerate(zip(potential1, density1)):
            e_kinetic -= vt.integrate(nt)
            if spin < density.ndensities:
                e_kinetic += vt.integrate(density.core_density)
                self.vt.data += vt.data / density.ndensities

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, potential1


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_pw: PlaneWaves,
                 fine_pw: PlaneWaves,
                 setups,
                 fracpos,
                 xc,
                 poisson_solver):
        self.vHt = fine_pw.zeros()  # initial guess for Coulomb potential
        self.nt = wf_pw.empty()
        self.vt = wf_pw.empty()

        self.vbar_acf = setups.create_local_potentials(wf_pw, fracpos)
        self.ghat_acf = setups.create_compensation_charges(fine_pw, fracpos)

        PotentialCalculator.__init__(self,xc, poisson_solver)

        self.pwmap = PWMapping(wf_pw, fine_pw)
        self.fftplan, self.ifftplan = wf_pw.grid.fft_plans()
        self.fftplan2, self.ifftplan2 = fine_pw.grid.fft_plans()

    def _calculate(self, density):
        fine_density = self.fine_grid.empty(density.shape)
        for spin, (nt1, nt2) in enumerate(zip(density.density, fine_density)):
            nt1.fft_interpolate(nt2, self.fftplan, self.ifftplan2)

        vxct = fine_density.grid.zeros(density.shape)
        e_xc = self.xc.calculate(fine_density, vxct)

        nt = density.grid.empty()
        nt.data[:] = density.data[:density.ndensities].sum(axis=0)
        nt.fft(plan=self.fftplan, out=self.nt)
        e_zero = self.vbar.integrate(self.nt)

        charge = self.vHt.pw.zeros()
        indices = self.pwmap.G2_G1
        scale = charge.pw.grid.size.prod() / self.nt.pw.grid.size.prod()
        assert scale == 8
        charge.data[indices] = self.nt.data * scale
        coefs = density.calculate_compensation_charge_coefficients()
        self.ghat_acf.add_to(charge, coefs)
        self.poisson_solver.solve(self.vHt, charge)
        e_coulomb = 0.5 * self.vHt.integrate(charge)

        vt = self.vbar.new()
        vt.data[:] = self.vbar.data
        vt.data += self.vHt.data[indices] * scale**-1
        potential1 = ...
        potential2 = vxct
        potential2.data += vt.ifft(plan=self.ifftplan).data

        e_kinetic = 0.0
        for spin, (vt, nt) in enumerate(zip(potential1, density1)):
            potential1 = self.restrict(potential2)
            e_kinetic -= p.integrate(d)
            if s < density.ndensities:
                e_kinetic += p.integrate(density.core_density)

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, potential1


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
