"""
==  ==========
R
r
G
g
h
x   r or h
==  ==========

"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from gpaw.core import PlaneWaves, UniformGrid
from gpaw.new.potential import Potential
from gpaw.new.xc import XCFunctional
from gpaw.setup import Setup
from gpaw.typing import Array1D, Array3D
from gpaw.utilities import pack, unpack


class PotentialCalculator:
    def __init__(self,
                 xc,
                 poisson_solver,
                 setups):
        self.poisson_solver = poisson_solver
        self.xc = xc
        self.setups = setups

    def __str__(self):
        return f'\n{self.poisson_solver}\n{self.xc}'

    def calculate(self, density, vHt_x=None):
        energies, vt_sR, vHt_x = self._calculate(density, vHt_x)

        Q_aL = self.calculate_charges(vHt_x)
        dH_asii, corrections = calculate_non_local_potential(
            self.setups, density, self.xc, Q_aL)

        for key, e in corrections.items():
            energies[key] += e

        return Potential(vt_sR, dH_asii, energies), vHt_x

    def move(self, fracpos_ac):
        self.nct_aR.move(fracpos_ac)
        self.vbar_ar.move(fracpos_ac)
        self.ghat_ar.move(fracpos_ac)
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)


class UniformGridPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_grid: UniformGrid,
                 fine_grid: UniformGrid,
                 setups,
                 xc,
                 poisson_solver,
                 nct_aR):
        self.nct_aR = nct_aR
        fracpos_ac = nct_aR.fracpos_ac
        self.vbar_ar = setups.create_local_potentials(fine_grid, fracpos_ac)
        self.ghat_aLr = setups.create_compensation_charges(fine_grid,
                                                           fracpos_ac)

        self.vbar_r = fine_grid.empty()
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)

        self.interpolate = wf_grid.transformer(fine_grid)
        self.restrict = fine_grid.transformer(wf_grid)

        super().__init__(xc, poisson_solver, setups)

    def calculate_charges(self, vHt_r):
        return self.ghat_aLr.integrate(vHt_r)

    def _calculate(self, density, vHt_r):
        nt_sR = density.nt_sR
        nt_sr = self.interpolate(nt_sR, preserve_integral=True)

        grid2 = nt_sr.desc

        vxct_sr = grid2.zeros(nt_sr.dims)
        e_xc = self.xc.calculate(nt_sr, vxct_sr)

        charge_r = grid2.empty()
        charge_r.data[:] = nt_sr.data[:density.ndensities].sum(axis=0)
        e_zero = self.vbar_r.integrate(charge_r)

        ccc_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_ar.add_to(charge_r, ccc_aL)
        if vHt_r is None:
            vHt_r = grid2.zeros()
        self.poisson_solver.solve(vHt_r, charge_r)
        e_coulomb = 0.5 * vHt_r.integrate(charge_r)

        vt_sr = vxct_sr
        vt_sr.data += vHt_r.data + self.vbar_r.data
        vt_sR = self.restrict(vt_sr)
        e_kinetic = 0.0
        for spin, (vt_R, nt_R) in enumerate(zip(vt_sR, nt_sR)):
            e_kinetic -= vt_R.integrate(nt_R)
            if spin < density.ndensities:
                e_kinetic += vt_R.integrate(density.nct_R)

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_r

    def force_contributions(self, state):
        density = state.density
        potential = state.potential
        nt_R = density.nt_sR[0]
        vt_R = potential.vt_sR[0]
        if density.ndensities > 1:
            nt_R = nt_R.desc.empty()
            nt_R.data[:] = density.nt_sR.data[:density.ndensities].sum(axis=0)
            vt_R = vt_R.desc.empty()
            vt_R.data[:] = (
                potential.vt_sR.data[:density.ndensities].sum(axis=0) /
                density.ndensities)

        nt_r = self.interpolate(nt_R, preserve_integral=True)

        return (self.ghat_ar.derivative(state.vHt_x),
                self.nct_aR.derivative(vt_R),
                self.vbar_ar.derivative(nt_r))


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 grid,
                 fine_grid,
                 pw: PlaneWaves,
                 fine_pw: PlaneWaves,
                 setups,
                 xc,
                 poisson_solver,
                 nct_ag):
        super().__init__(xc, poisson_solver, setups)

        fracpos_ac = nct_ag.fracpos_ac
        self.nct_ag = nct_ag
        self.vbar_ag = setups.create_local_potentials(pw, fracpos_ac)
        self.ghat_aLh = setups.create_compensation_charges(fine_pw, fracpos_ac)

        self.h_g = fine_pw.map_indices(pw)
        self.fftplan, self.ifftplan = grid.fft_plans()
        self.fftplan2, self.ifftplan2 = fine_grid.fft_plans()
        self.fine_grid = fine_grid

        self.vbar_g = pw.zeros()
        self.vbar_ag.add_to(self.vbar_g)

    def calculate_charges(self, vHt_h):
        return self.ghat_aLh.integrate(vHt_h)

    def _calculate(self, density, vHt_h):
        nt_sr = self.fine_grid.empty(density.nt_sR.dims)
        nt_g = self.vbar_g.desc.zeros()
        indices = nt_g.desc.indices(self.fftplan.out_R.shape)
        for spin, (nt_R, nt_r) in enumerate(zip(density.nt_sR, nt_sr)):
            nt_R.fft_interpolate(nt_r, self.fftplan, self.ifftplan2)
            if spin < density.ndensities:
                nt_g.data += self.fftplan.out_R.ravel()[indices]

        e_zero = self.vbar_g.integrate(nt_g)

        if vHt_h is None:
            vHt_h = self.ghat_aLh.pw.zeros()

        charge_h = vHt_h.desc.zeros()
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)
        scale = 8
        charge_h.data[self.h_g] += nt_g.data * scale
        # background charge ???

        self.poisson_solver.solve(vHt_h, charge_h)
        e_coulomb = 0.5 * vHt_h.integrate(charge_h)

        vt_g = self.vbar_g.copy()
        vt_g.data += vHt_h.data[self.h_g] * scale**-1

        vt_sR = density.nt_sR.new()
        vt_sR.data[:] = vt_g.ifft(self.ifftplan, grid=vt_sR.desc).data
        vxct_sr = nt_sr.desc.zeros(density.nt_sR.dims)
        e_xc = self.xc.calculate(nt_sr, vxct_sr)

        vtmp_R = vt_sR.desc.empty()
        e_kinetic = 0.0
        for spin, (vt_R, vxct_r) in enumerate(zip(vt_sR, vxct_sr)):
            vxct_r.fft_restrict(vtmp_R, self.fftplan2, self.ifftplan)
            vt_R.data += vtmp_R.data
            e_kinetic -= vt_R.integrate(density.nt_sR[spin])
            if spin < density.ndensities:
                e_kinetic += vt_R.integrate(density.nct_R)

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_h

    def forces(self, nct_ag):
        return (self.ghat_ah.derivative(self.vHt_h),
                nct_ag.derivative(self.vt_g),
                self.vbar_ag.derivative(self.nt_g))


def calculate_non_local_potential(setups,
                                  density,
                                  xc,
                                  Q_aL):
    dH_asii = density.D_asii.new()
    energy_corrections = defaultdict(float)
    for a, D_sii in density.D_asii.items():
        Q_L = Q_aL[a]
        setup = setups[a]
        dH_sii, energies = calculate_non_local_potential1(
            setup, xc, D_sii, Q_L)
        dH_asii[a][:] = dH_sii
        for key, e in energies.items():
            energy_corrections[key] += e

    # Sum over domain:
    energies = np.array(list(energy_corrections.values()))
    density.D_asii.layout.atomdist.comm.sum(energies)
    energy_corrections = {name: e for name, e in zip(energy_corrections,
                                                     energies)}
    return dH_asii, energy_corrections


def calculate_non_local_potential1(setup: Setup,
                                   xc: XCFunctional,
                                   D_sii: Array3D,
                                   Q_L: Array1D) -> tuple[Array3D,
                                                          dict[str, float]]:
    ndensities = 2 if len(D_sii) == 2 else 1
    D_sp = np.array([pack(D_ii) for D_ii in D_sii])

    D_p = D_sp[:ndensities].sum(0)

    dH_p = (setup.K_p + setup.M_p +
            setup.MB_p + 2.0 * setup.M_pp @ D_p +
            setup.Delta_pL @ Q_L)
    e_kinetic = setup.K_p @ D_p + setup.Kc
    e_zero = setup.MB + setup.MB_p @ D_p
    e_coulomb = setup.M + D_p @ (setup.M_p + setup.M_pp @ D_p)

    dH_sp = np.zeros_like(D_sp)
    dH_sp[:ndensities] = dH_p
    e_xc = xc.calculate_paw_correction(setup, D_sp, dH_sp)
    e_kinetic -= (D_sp * dH_sp).sum().real

    e_external = 0.0

    dH_sii = unpack(dH_sp)

    return dH_sii, {'kinetic': e_kinetic,
                    'coulomb': e_coulomb,
                    'zero': e_zero,
                    'xc': e_xc,
                    'external': e_external}
