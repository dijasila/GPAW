from __future__ import annotations
from collections import defaultdict
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

    def __str__(self):
        return f'\n{self.poisson_solver}\n{self.xc}'

    def calculate(self, density):
        energies, potential = self._calculate(density)

        vnonloc, corrections = calculate_non_local_potential(
            density, self.xc, self.ghat_acf, self.vHt)

        for key, e in corrections.items():
            energies[key] += e

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

        self.vbar = self.vbar_acf.to_uniform_grid()

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
                 pw: PlaneWaves,
                 fine_pw: PlaneWaves,
                 setups,
                 fracpos,
                 xc,
                 poisson_solver):
        self.vHt = fine_pw.zeros()  # initial guess for Coulomb potential
        self.nt = pw.empty()
        self.vt = pw.empty()

        self.vbar_acf = setups.create_local_potentials(pw, fracpos)
        self.ghat_acf = setups.create_compensation_charges(fine_pw, fracpos)

        PotentialCalculator.__init__(self, xc, poisson_solver)

        self.pwmap = PWMapping(pw, fine_pw)
        self.fftplan, self.ifftplan = pw.grid.fft_plans()
        self.fftplan2, self.ifftplan2 = fine_pw.grid.fft_plans()

        self.fine_grid = fine_pw.grid

        self.vbar = pw.zeros()
        self.vbar_acf.add_to(self.vbar)

    def _calculate(self, density):
        fine_density = self.fine_grid.empty(density.density.shape)
        self.nt.data[:] = 0.0
        for spin, (nt1, nt2) in enumerate(zip(density.density, fine_density)):
            nt1.fft_interpolate(nt2, self.fftplan, self.ifftplan2)
            if spin < density.ndensities:
                self.nt.data += self.fftplan.out_R.ravel()[self.nt.pw.indices]

        e_zero = self.vbar.integrate(self.nt)

        charge = self.vHt.pw.zeros()
        coefs = density.calculate_compensation_charge_coefficients()
        self.ghat_acf.add_to(charge, coefs)
        indices = self.pwmap.G2_G1
        scale = charge.pw.grid.size.prod() / self.nt.pw.grid.size.prod()
        assert scale == 8
        charge.data[indices] += self.nt.data * scale
        # background charge ???

        self.poisson_solver.solve(self.vHt, charge)
        e_coulomb = 0.5 * self.vHt.integrate(charge)

        self.vt.data[:] = self.vbar.data
        self.vt.data += self.vHt.data[indices] * scale**-1

        potential = density.density.new()
        potential.data[:] = self.vt.ifft().data
        vxct = fine_density.grid.zeros(density.density.shape)
        e_xc = self.xc.calculate(fine_density, vxct)

        vtmp = potential.grid.empty()
        e_kinetic = 0.0
        for spin, (vt1, vxct_fine) in enumerate(zip(potential, vxct)):
            vxct_fine.fft_restrict(vtmp, self.fftplan2, self.ifftplan)
            vt1.data += vtmp.data
            e_kinetic -= vt1.integrate(density.density[spin])
            if spin < density.ndensities:
                self.vt.data += (self.fftplan2.out_R.ravel()[indices] /
                                 density.ndensities)
                e_kinetic -= vt1.integrate(density.core_density)

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, potential


def calculate_non_local_potential(density, xc,
                                  ghat_acf, vHt):
    coefs = ghat_acf.integrate(vHt)
    vnonloc = density.density_matrices.new()
    energy_corrections = defaultdict(float)
    for a, D in density.density_matrices.items():
        Q = coefs[a]
        setup = density.setups[a]
        dH, energies = calculate_non_local_potential1(setup, xc, D, Q)
        vnonloc[a][:] = dH
        for key, e in energies.items():
            energy_corrections[key] += e

    return vnonloc, energy_corrections


def calculate_non_local_potential1(setup: Setup,
                                   xc: XCFunctional,
                                   D: Array3D,
                                   Q: Array1D) -> tuple[Array3D,
                                                        dict[str, float]]:
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

    return H, {'kinetic': e_kinetic,
               'coulomb': e_coulomb,
               'zero': e_zero,
               'xc': e_xc,
               'external': e_external}
