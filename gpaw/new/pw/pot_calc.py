class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 grid,
                 fine_grid,
                 pw: PlaneWaves,
                 fine_pw: PlaneWaves,
                 setups,
                 xc,
                 poisson_solver,
                 nct_ag,
                 nct_R):
        super().__init__(xc, poisson_solver, setups, nct_R)

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
        nt_g.data *= 1 / self.fftplan.in_R.size

        e_zero = self.vbar_g.integrate(nt_g)

        if vHt_h is None:
            vHt_h = self.ghat_aLh.pw.zeros()

        charge_h = vHt_h.desc.zeros()
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)
        charge_h.data[self.h_g] += nt_g.data
        # background charge ???

        self.poisson_solver.solve(vHt_h, charge_h)
        e_coulomb = 0.5 * vHt_h.integrate(charge_h)

        vt_g = self.vbar_g.copy()
        vt_g.data += vHt_h.data[self.h_g]

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
                e_kinetic += vt_R.integrate(self.nct_R)

        e_external = 0.0

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_h

    def _move_nct(self, fracpos_ac, ndensities):
        self.ghat_aLr.move(fracpos_ac)
        self.vbar_ar.move(fracpos_ac)
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)
        self.nct_aR.move(fracpos_ac)
        self.nct_aR.to_uniform_grid(out=self.nct_R, scale=1.0 / ndensities)

    def forces(self, nct_ag):
        return (self.ghat_ah.derivative(self.vHt_h),
                nct_ag.derivative(self.vt_g),
                self.vbar_ag.derivative(self.nt_g))
