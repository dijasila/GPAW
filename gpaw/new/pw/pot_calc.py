import numpy as np
from gpaw.core import PlaneWaves
from gpaw.new.pot_calc import PotentialCalculator


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
                 nct_R,
                 soc=False):
        fracpos_ac = nct_ag.fracpos_ac
        atomdist = nct_ag.atomdist
        super().__init__(xc, poisson_solver, setups, nct_R, fracpos_ac, soc)

        self.nct_ag = nct_ag
        self.vbar_ag = setups.create_local_potentials(pw, fracpos_ac, atomdist)
        self.ghat_aLh = setups.create_compensation_charges(
            fine_pw, fracpos_ac, atomdist)

        self.pw = pw
        self.fine_pw = fine_pw
        self.pw0 = pw.new(comm=None)  # not distributed

        self.h_g, self.g_r = fine_pw.map_indices(self.pw0)

        self.fftplan = grid.fft_plans()
        self.fftplan2 = fine_grid.fft_plans()

        self.fine_grid = fine_grid

        self.vbar_g = pw.zeros()
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()

        self._nt_g = None
        self._vt_g = None

    def calculate_charges(self, vHt_h):
        return self.ghat_aLh.integrate(vHt_h)

    def calculate_non_selfconsistent_exc(self, nt_sR, xc):
        nt_sr, _, _ = self._interpolate_density(nt_sR)
        vxct_sr = nt_sr.desc.zeros(nt_sr.dims)
        e_xc = xc.calculate(nt_sr, vxct_sr)
        return e_xc

    def _interpolate_density(self, nt_sR):
        nt_sr = self.fine_grid.empty(nt_sR.dims)
        pw = self.vbar_g.desc

        if pw.comm.rank == 0:
            indices = self.pw0.indices(self.fftplan.tmp_Q.shape)
            nt0_g = self.pw0.zeros()
        else:
            nt0_g = None

        ndensities = nt_sR.dims[0] % 3
        for spin, (nt_R, nt_r) in enumerate(zip(nt_sR, nt_sr)):
            nt_R.interpolate(self.fftplan, self.fftplan2, out=nt_r)
            if spin < ndensities and pw.comm.rank == 0:
                nt0_g.data += self.fftplan.tmp_Q.ravel()[indices]

        return nt_sr, pw, nt0_g

    def _calculate(self, density, vHt_h):
        #print(np.shape(density.nt_sR.data))
        nt_sr, pw, nt0_g = self._interpolate_density(density.nt_sR)
        #with open('1.txt', 'w') as fd:
        #    fd.write(density.nt_sR.data)
        #input('')
        vxct_sr = nt_sr.desc.zeros(density.nt_sR.dims)
        e_xc = self.xc.calculate(nt_sr, vxct_sr)
        


        if pw.comm.rank == 0:
            nt0_g.data *= 1 / np.prod(density.nt_sR.desc.size_c)
            e_zero = self.vbar0_g.integrate(nt0_g)
        else:
            e_zero = 0.0
        e_zero = pw.comm.sum(e_zero)  # use broadcast XXX

        if vHt_h is None:
            vHt_h = self.ghat_aLh.pw.zeros()

        charge_h = vHt_h.desc.zeros()
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)

        if pw.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt0_g.data[g]
                else:
                    pw.comm.send(nt0_g.data[g], rank)
        else:
            data = np.empty(len(self.h_g), complex)
            pw.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        # background charge ???

        self.poisson_solver.solve(vHt_h, charge_h)
        e_coulomb = 0.5 * vHt_h.integrate(charge_h)

        if pw.comm.rank == 0:
            vt0_g = self.vbar0_g.copy()
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = np.empty(len(g), complex)
                    pw.comm.receive(data, rank)
                    vt0_g.data[g] += data
            vt0_R = vt0_g.ifft(plan=self.fftplan,
                               grid=density.nt_sR.desc.new(comm=None))
        else:
            pw.comm.send(vHt_h.data[self.h_g], 0)

        vt_sR = density.nt_sR.new()
        vt_sR[0].scatter_from(vt0_R if pw.comm.rank == 0 else None)
        if density.ndensities == 2:
            vt_sR.data[1] = vt_sR.data[0]
        vt_sR.data[density.ndensities:] = 0.0

        e_kinetic = self._restrict(vxct_sr, vt_sR, density)

        e_external = 0.0

        self._reset()

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_h

    def _restrict(self, vxct_sr, vt_sR, density=None):
        vtmp_R = vt_sR.desc.empty()
        e_kinetic = 0.0
        for spin, (vt_R, vxct_r) in enumerate(zip(vt_sR, vxct_sr)):
            vxct_r.fft_restrict(self.fftplan2, self.fftplan, out=vtmp_R)
            vt_R.data += vtmp_R.data
            if density:
                e_kinetic -= vt_R.integrate(density.nt_sR[spin])
                if spin < density.ndensities:
                    e_kinetic += vt_R.integrate(self.nct_R)
        return e_kinetic

    def _move_nct(self, fracpos_ac, ndensities):
        self.ghat_aLh.move(fracpos_ac)
        self.vbar_ar.move(fracpos_ac)
        self.vbar_ar.to_uniform_grid(out=self.vbar_r)
        self.vbar0_g = self.vbar_g.gather()
        self.nct_aR.move(fracpos_ac)
        self.nct_aR.to_uniform_grid(out=self.nct_R, scale=1.0 / ndensities)
        self._reset()

    def _reset(self):
        self._vt_g = None
        self._nt_g = None

    def _force_stress_helper(self, state):
        if self._vt_g is None:
            density = state.density
            potential = state.potential
            nt_R = density.nt_sR[0]
            vt_R = potential.vt_sR[0]
            if density.ndensities > 1:
                nt_R = nt_R.desc.empty()
                nt_R.data[:] = density.nt_sR.data[:density.ndensities].sum(
                    axis=0)
                vt_R = vt_R.desc.empty()
                vt_R.data[:] = (
                    potential.vt_sR.data[:density.ndensities].sum(axis=0) /
                    density.ndensities)
            self._vt_g = vt_R.fft(self.fftplan, pw=self.pw)
            self._nt_g = nt_R.fft(self.fftplan, pw=self.pw)
        return self._vt_g, self._nt_g

    def force_contributions(self, state):
        vt_g, nt_g = self._force_stress_helper(state)

        return (self.ghat_aLh.derivative(state.vHt_x),
                self.nct_ag.derivative(vt_g),
                self.vbar_ag.derivative(nt_g))

    def stress_contributions(self, state):
        vt_g, nt_g = self._force_stress_helper(state)
        return ...
