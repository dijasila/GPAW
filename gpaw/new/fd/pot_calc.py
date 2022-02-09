from gpaw.new.pot_calc import PotentialCalculator
from gpaw.core import UniformGrid


class UniformGridPotentialCalculator(PotentialCalculator):
    def __init__(self,
                 wf_grid: UniformGrid,
                 fine_grid: UniformGrid,
                 setups,
                 xc,
                 poisson_solver,
                 nct_aR, nct_R,
                 interpolation_stencil_range=3):
        self.nct_aR = nct_aR

        fracpos_ac = nct_aR.fracpos_ac

        self.vbar_ar = setups.create_local_potentials(fine_grid, fracpos_ac)
        self.ghat_aLr = setups.create_compensation_charges(fine_grid,
                                                           fracpos_ac)

        self.vbar_r = fine_grid.empty()
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)

        n = interpolation_stencil_range
        self.interpolate = wf_grid.transformer(fine_grid, n)
        self.restrict = fine_grid.transformer(wf_grid, n)

        super().__init__(xc, poisson_solver, setups, nct_R)

    def calculate_charges(self, vHt_r):
        return self.ghat_aLr.integrate(vHt_r)

    def _calculate(self, density, vHt_r):
        nt_sR = density.nt_sR
        nt_sr = self.interpolate(nt_sR)
        if not nt_sR.desc.pbc_c.all():
            Nt1_s = nt_sR.integrate()
            Nt2_s = nt_sr.integrate()
            for Nt1, Nt2, nt_r in zip(Nt1_s, Nt2_s, nt_sr):
                if Nt2 > 1e-14:
                    nt_r.data *= Nt1 / Nt2
        grid2 = nt_sr.desc

        vxct_sr = grid2.zeros(nt_sr.dims)
        e_xc = self.xc.calculate(nt_sr, vxct_sr)

        charge_r = grid2.empty()
        charge_r.data[:] = nt_sr.data[:density.ndensities].sum(axis=0)
        e_zero = self.vbar_r.integrate(charge_r)

        ccc_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLr.add_to(charge_r, ccc_aL)
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
                e_kinetic += vt_R.integrate(self.nct_R)

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_r

    def _move(self, fracpos_ac, ndensities):
        self.ghat_aLr.move(fracpos_ac)
        self.vbar_ar.move(fracpos_ac)
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)
        self.nct_aR.move(fracpos_ac)
        self.nct_aR.to_uniform_grid(out=self.nct_R, scale=1.0 / ndensities)

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

        nt_r = self.interpolate(nt_R)
        if not nt_r.desc.pbc_c.all():
            scale = nt_R.integrate() / nt_r.integrate()
            nt_r.data *= scale

        return (self.ghat_aLr.derivative(state.vHt_x),
                self.nct_aR.derivative(vt_R),
                self.vbar_ar.derivative(nt_r))
