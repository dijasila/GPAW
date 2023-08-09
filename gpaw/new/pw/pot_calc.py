import numpy as np
from gpaw.core import PlaneWaves
from gpaw.gpu import cupy as cp
from gpaw.mpi import broadcast_float
from gpaw.new import zip
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.stress import calculate_stress
from gpaw.setup import Setups


class PlaneWavePotentialCalculator(PotentialCalculator):
    def __init__(self,
                 grid,
                 fine_grid,
                 pw: PlaneWaves,
                 fine_pw: PlaneWaves,
                 setups: Setups,
                 xc,
                 poisson_solver,
                 nct_ag,
                 nct_R,
                 soc=False,
                 xp=np):
        fracpos_ac = nct_ag.fracpos_ac
        atomdist = nct_ag.atomdist
        self.xp = xp
        super().__init__(xc, poisson_solver, setups, nct_R, fracpos_ac, soc)

        self.nct_ag = nct_ag
        self.vbar_ag = setups.create_local_potentials(
            pw, fracpos_ac, atomdist, xp)
        self.ghat_aLh = setups.create_compensation_charges(
            fine_pw, fracpos_ac, atomdist, xp)

        self.pw = pw
        self.fine_pw = fine_pw
        self.pw0 = pw.new(comm=None)  # not distributed

        self.h_g, self.g_r = fine_pw.map_indices(self.pw0)
        if xp is cp:
            self.h_g = cp.asarray(self.h_g)
            self.g_r = [cp.asarray(g) for g in self.g_r]

        self.fftplan = grid.fft_plans(xp=xp)
        self.fftplan2 = fine_grid.fft_plans(xp=xp)

        self.grid = grid
        self.fine_grid = fine_grid

        self.vbar_g = pw.zeros(xp=xp)
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()

        self._nt_g = None
        self._vt_g = None

        self.e_stress = np.nan

        self.interpolation_domain = nct_ag.pw

    def interpolate(self, a_R, a_r):
        a_R.interpolate(self.fftplan, self.fftplan2, out=a_r)

    def restrict(self, a_r, a_R):
        a_r.fft_restrict(self.fftplan2, self.fftplan, out=a_R)

    def calculate_charges(self, vHt_h):
        return self.ghat_aLh.integrate(vHt_h)

    def _interpolate_density(self, nt_sR):
        nt_sr = self.fine_grid.empty(nt_sR.dims, xp=self.xp)
        pw = self.vbar_g.desc

        if pw.comm.rank == 0:
            indices = self.xp.asarray(self.pw0.indices(self.fftplan.shape))
            nt0_g = self.pw0.zeros(xp=self.xp)
        else:
            nt0_g = None

        ndensities = nt_sR.dims[0] % 3
        for spin, (nt_R, nt_r) in enumerate(zip(nt_sR, nt_sr)):
            self.interpolate(nt_R, nt_r)
            if spin < ndensities and pw.comm.rank == 0:
                nt0_g.data += self.xp.asarray(
                    self.fftplan.tmp_Q.ravel()[indices])

        return nt_sr, pw, nt0_g

    def _interpolate_and_calculate_xc(self, xc, nt_sR, ibzwfs):
        nt_sr, pw, nt0_g = self._interpolate_density(nt_sR)
        vxct_sr = nt_sr.desc.empty(nt_sR.dims, xp=self.xp)
        e_xc, e_kinetic = self.xc.calculate(
            nt_sr, vxct_sr, ibzwfs,
            interpolate=self.interpolate,
            restrict=self.restrict)
        return nt_sr, pw, nt0_g, vxct_sr, e_xc, e_kinetic

    def calculate_non_selfconsistent_exc(self, xc, nt_sR, ibzwfs):
        _, _, _, _, e_xc, _ = self._interpolate_and_calculate_xc(
            xc, nt_sR, ibzwfs)
        return e_xc

    def calculate_pseudo_potential(self, density, ibzwfs, vHt_h):
        nt_sr, pw, nt0_g, vxct_sr, e_xc, e_kinetic = (
            self._interpolate_and_calculate_xc(
                self.xc, density.nt_sR, ibzwfs))

        if pw.comm.rank == 0:
            nt0_g.data *= 1 / np.prod(density.nt_sR.desc.size_c)
            e_zero = self.vbar0_g.integrate(nt0_g)
        else:
            e_zero = 0.0
        e_zero = broadcast_float(float(e_zero), pw.comm)

        if vHt_h is None:
            vHt_h = self.ghat_aLh.pw.zeros(xp=self.xp)

        charge_h = vHt_h.desc.zeros(xp=self.xp)
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)

        if pw.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt0_g.data[g]
                else:
                    pw.comm.send(nt0_g.data[g], rank)
        else:
            data = self.xp.empty(len(self.h_g), complex)
            pw.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        # background charge ???

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)

        if pw.comm.rank == 0:
            vt0_g = self.vbar0_g.copy()
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = self.xp.empty(len(g), complex)
                    pw.comm.receive(data, rank)
                    vt0_g.data[g] += data
            vt0_R = vt0_g.ifft(
                plan=self.fftplan,
                grid=density.nt_sR.desc.new(comm=None))
        else:
            pw.comm.send(vHt_h.data[self.h_g], 0)

        vt_sR = density.nt_sR.new()
        vt_sR[0].scatter_from(vt0_R if pw.comm.rank == 0 else None)
        if density.ndensities == 2:
            vt_sR.data[1] = vt_sR.data[0]
        vt_sR.data[density.ndensities:] = 0.0

        e_kinetic += self._restrict(vxct_sr, vt_sR, density)

        e_external = 0.0

        self.e_stress = e_coulomb + e_zero

        self._reset()

        return {'kinetic': e_kinetic,
                'coulomb': e_coulomb,
                'zero': e_zero,
                'xc': e_xc,
                'external': e_external}, vt_sR, vHt_h

    def _restrict(self, vxct_sr, vt_sR, density=None):
        vtmp_R = vt_sR.desc.empty(xp=self.xp)
        e_kinetic = 0.0
        for spin, (vt_R, vxct_r) in enumerate(zip(vt_sR, vxct_sr)):
            self.restrict(vxct_r, vtmp_R)
            vt_R.data += vtmp_R.data
            if density:
                e_kinetic -= vt_R.integrate(density.nt_sR[spin])
                if spin < density.ndensities:
                    e_kinetic += vt_R.integrate(self.nct_R)
        return float(e_kinetic)

    def xxxrestrict(self, vt_sr):
        vt_sR = self.grid.empty(vt_sr.dims, xp=self.xp)
        for vt_R, vt_r in zip(vt_sR, vt_sr):
            vt_r.fft_restrict(
                self.fftplan2, self.fftplan, out=vt_R)
        return vt_sR

    def _move(self, fracpos_ac, atomdist, ndensities):
        self.ghat_aLh.move(fracpos_ac, atomdist)
        self.vbar_ag.move(fracpos_ac, atomdist)
        self.vbar_g.data[:] = 0.0
        self.vbar_ag.add_to(self.vbar_g)
        self.vbar0_g = self.vbar_g.gather()
        self.nct_ag.move(fracpos_ac, atomdist)
        self.nct_ag.to_uniform_grid(out=self.nct_R, scale=1.0 / ndensities)
        self.xc.move(fracpos_ac, atomdist)
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
                nt_R = nt_R.desc.empty(xp=self.xp)
                nt_R.data[:] = density.nt_sR.data[:density.ndensities].sum(
                    axis=0)
                vt_R = vt_R.desc.empty(xp=self.xp)
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

    def stress(self, state):
        vt_g, nt_g = self._force_stress_helper(state)
        return calculate_stress(self, state, vt_g, nt_g)
