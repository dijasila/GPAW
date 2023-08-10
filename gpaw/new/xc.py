from __future__ import annotations

import numpy as np

import _gpaw
from gpaw.fd_operators import Gradient
from gpaw.new import zips
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional as OldXCFunctional
from gpaw.xc.gga import add_gradient_correction, gga_vars


def create_functional(xc: OldXCFunctional | str | dict,
                      grid, coarse_grid, interpolate_domain,
                      setups, fracpos_ac, atomdist):
    if isinstance(xc, (str, dict)):
        xc = XC(xc)
    if xc.type == 'MGGA':
        return MGGAFunctional(xc, grid, coarse_grid,
                              interpolate_domain, setups, fracpos_ac, atomdist)
    return LDAOrGGAFunctional(xc, grid)


class Functional:
    def __init__(self, xc, grid):
        self.xc = xc
        self.grid = grid
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.no_forces = self.name.startswith('GLLB')
        self.type = self.xc.type
        self.xc.set_grid_descriptor(grid._gd)

    def __str__(self):
        return f'name: {self.xc.get_description()}'

    def calculate_paw_correction(self, setup, d, h=None):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self) -> str:
        return self.name

    def apply(self, spin, psit_nX, vt_nX) -> None:
        pass

    def move(self, fracpos_ac, atomdist) -> None:
        pass


class LDAOrGGAFunctional(Functional):
    def calculate(self,
                  nt_sr,
                  vxct_sr,
                  ibzwfs=None,
                  interpolate=None,
                  restrict=None) -> tuple[float, float]:
        if nt_sr.xp is np:
            vxct_sr.data[:] = 0.0
            return self.xc.calculate(self.xc.gd, nt_sr.data, vxct_sr.data), 0.0
        vxct_np_sr = np.zeros(vxct_sr.data.shape)
        exc = self.xc.calculate(nt_sr.desc._gd, nt_sr.data.get(), vxct_np_sr)
        vxct_sr.data[:] = vxct_sr.xp.asarray(vxct_np_sr)
        return exc, 0.0


class MGGAFunctional(Functional):
    def __init__(self,
                 xc,
                 grid,
                 coarse_grid,
                 domain,
                 setups,
                 fracpos_ac,
                 atomdist):
        super().__init__(xc, grid)
        tauct_aX = setups.create_pseudo_core_kinetic_energy_densities(
            domain, fracpos_ac, atomdist)
        self.ked_calculator = KEDCalculator.create(tauct_aX)
        self.coarse_grid = coarse_grid
        self.dedtaut_sR = None

    def move(self, fracpos_ac, atomdist):
        self.ked_calculator.move(fracpos_ac, atomdist)

    def get_setup_name(self):
        return 'PBE'

    def apply(self, spin, psit_nX, vt_nX):
        self.ked_calculator._apply(self.dedtaut_sR[spin], psit_nX, vt_nX)

    def calculate(self,
                  nt_sr,
                  vxct_sr,
                  ibzwfs,
                  interpolate,
                  restrict) -> tuple[float, float]:
        nspins = nt_sr.dims[0]
        taut_sR = self.coarse_grid.empty(nspins)
        self.ked_calculator.calculate_pseudo_ked(ibzwfs, taut_sR)
        taut_sr = self.grid.empty(nspins)
        interpolate(taut_sR, taut_sr)

        gd = self.xc.gd

        sigma_xr, dedsigma_xr, gradn_svr = gga_vars(gd, self.xc.grad_v,
                                                    nt_sr.data)

        e_r = self.grid.empty()
        dedtaut_sr = taut_sr.new()
        vxct_sr.data[:] = 0.0
        self.xc.kernel.calculate(e_r.data, nt_sr.data, vxct_sr.data,
                                 sigma_xr, dedsigma_xr,
                                 taut_sr.data, dedtaut_sr.data)

        self.dedtaut_sR = taut_sR.new()
        ekin = 0.0
        for dedtaut_R, dedtaut_r, taut_R in zips(self.dedtaut_sR,
                                                 dedtaut_sr,
                                                 taut_sR):
            restrict(dedtaut_r, dedtaut_R)
            taut_R.data -= self.ked_calculator.tauct_R.data
            ekin -= dedtaut_R.integrate(taut_R)

        add_gradient_correction(self.xc.grad_v, gradn_svr, sigma_xr,
                                dedsigma_xr, vxct_sr.data)

        return e_r.integrate(), ekin


class KEDCalculator:
    @staticmethod
    def create(tauct_aX):
        if hasattr(tauct_aX, 'pw'):
            return PWKEDCalculator(tauct_aX)
        return FDKEDCalculator(tauct_aX)

    def __init__(self, tauct_aX):
        self.tauct_aX = tauct_aX
        self.tauct_R = None

    def move(self, fracpos_ac, atomdist):
        self.tauct_aX.move(fracpos_ac, atomdist)
        self.tauct_R = None

    def calculate_pseudo_ked(self, ibzwfs, taut_sR):
        nspins = taut_sR.dims[0]
        if self.tauct_R is None:
            self.tauct_R = taut_sR.desc.empty()
            self.tauct_aX.to_uniform_grid(out=self.tauct_R,
                                          scale=1.0 / nspins)

        taut_sR.data[:] = 0.0

        if ibzwfs is not None:
            for wfs in ibzwfs:
                occ_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n
                self.add_ked(occ_n, wfs.psit_nX, taut_sR[wfs.spin])
            taut_sR.symmetrize(ibzwfs.ibz.symmetries.rotation_scc,
                               ibzwfs.ibz.symmetries.translation_sc)
            ibzwfs.kpt_comm.sum(taut_sR.data)
            ibzwfs.band_comm.sum(taut_sR.data)

        taut_sR.data += self.tauct_R.data

    def add_ked(self, occ_n, psit_nX, taut_R):
        raise NotImplementedError


class PWKEDCalculator(KEDCalculator):
    def add_ked(self, occ_n, psit_nX, taut_R):
        psit_nG = psit_nX
        pw = psit_nG.desc
        domain_comm = pw.comm

        # Undistributed work arrays:
        dpsit1_R = taut_R.desc.new(comm=None, dtype=pw.dtype).empty()
        pw1 = pw.new(comm=None)
        psit1_G = pw1.empty()
        iGpsit1_G = pw1.empty()
        taut1_R = taut_R.desc.new(comm=None).zeros()
        Gplusk1_Gv = pw1.reciprocal_vectors()

        (N,) = psit_nG.mydims
        for n1 in range(0, N, domain_comm.size):
            n2 = min(n1 + domain_comm.size, N)
            psit_nG[n1:n2].gather_all(psit1_G)
            n = n1 + domain_comm.rank
            if n >= N:
                continue
            f = occ_n[n]
            if f == 0.0:
                continue
            for v in range(3):
                iGpsit1_G.data[:] = psit1_G.data
                iGpsit1_G.data *= 1j * Gplusk1_Gv[:, v]
                iGpsit1_G.ifft(out=dpsit1_R)
                # taut1_R.data += 0.5 * f * abs(dpsit1_R.data)**2
                _gpaw.add_to_density(0.5 * f, dpsit1_R.data, taut1_R.data)

        domain_comm.sum(taut1_R.data)
        tmp_R = taut_R.new()
        tmp_R.scatter_from(taut1_R)
        taut_R.data += tmp_R.data

    def _apply(self, dedtaut_R, psit_nG, vt_nG):
        pw = psit_nG.desc
        dpsit_R = dedtaut_R.desc.new(dtype=pw.dtype).empty()
        Gplusk1_Gv = pw.reciprocal_vectors()
        tmp_G = pw.empty()

        for psit_G, vt_G in zips(psit_nG, vt_nG):
            for v in range(3):
                tmp_G.data[:] = psit_G.data
                tmp_G.data *= 1j * Gplusk1_Gv[:, v]
                tmp_G.ifft(out=dpsit_R)
                dpsit_R.data *= dedtaut_R.data
                dpsit_R.fft(out=tmp_G)
                vt_G.data -= 0.5j * Gplusk1_Gv[:, v] * tmp_G.data


class FDKEDCalculator(KEDCalculator):
    def __init__(self, tauct_aX):
        super().__init__(tauct_aX)
        self.grad_v = []

    def add_ked(self, occ_n, psit_nX, taut_R):
        psit_nR = psit_nX
        if len(self.grad_v) == 0:
            grid = psit_nR.desc
            self.grad_v = [
                Gradient(grid._gd, v, n=3, dtype=grid.dtype)
                for v in range(3)]

        tmp_R = psit_nR.desc.empty()
        for f, psit_R in zips(occ_n, psit_nR):
            for grad in self.grad_v:
                grad(psit_R, tmp_R)
                # taut_R.data += 0.5 * f * abs(tmp_R.data)**2
                _gpaw.add_to_density(0.5 * f, tmp_R.data, taut_R.data)
