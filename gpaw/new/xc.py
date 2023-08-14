from __future__ import annotations

from typing import Callable

import numpy as np
from gpaw.core.plane_waves import PlaneWaveExpansions as PWArray
from gpaw.core.uniform_grid import UniformGrid as UGType
from gpaw.core.uniform_grid import UniformGridFunctions as UGArray
from gpaw.fd_operators import Gradient
from gpaw.new import zips
from gpaw.new.c import add_to_density
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array1D
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional as OldXCFunctional
from gpaw.xc.gga import add_gradient_correction, gga_vars
from gpaw.xc.mgga import MGGA
from gpaw.gpu import cupy as cp


def create_functional(xc: OldXCFunctional | str | dict,
                      grid: UGType):
    if isinstance(xc, (str, dict)):
        xc = XC(xc)
    if xc.type == 'MGGA':
        return MGGAFunctional(xc, grid)
    assert xc.type in {'LDA', 'GGA'}, xc
    return LDAOrGGAFunctional(xc, grid)


class Functional:
    def __init__(self,
                 xc: OldXCFunctional,
                 grid: UGType):
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


class LDAOrGGAFunctional(Functional):
    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        xp = nt_sr.xp
        vxct_sr = nt_sr.new()
        if xp is np:
            vxct_sr.data[:] = 0.0
            exc = self.xc.calculate(self.xc.gd, nt_sr.data, vxct_sr.data)
        else:
            vxct_np_sr = np.zeros(nt_sr.data.shape)
            assert isinstance(nt_sr.data, cp.ndarray)
            exc = self.xc.calculate(nt_sr.desc._gd, nt_sr.data.get(),
                                    vxct_np_sr)
            vxct_sr.data[:] = xp.asarray(vxct_np_sr)
        return exc, vxct_sr, None


class MGGAFunctional(Functional):
    def get_setup_name(self):
        return 'PBE'

    def calculate(self,
                  nt_sr,
                  taut_sr) -> tuple[float, UGArray, UGArray | None]:
        gd = self.xc.gd
        assert isinstance(self.xc, MGGA), self.xc
        sigma_xr, dedsigma_xr, gradn_svr = gga_vars(gd, self.xc.grad_v,
                                                    nt_sr.data)
        e_r = self.grid.empty()
        if taut_sr is None:
            taut_sr = nt_sr.new(zeroed=True)
        dedtaut_sr = taut_sr.new()
        vxct_sr = taut_sr.new()
        vxct_sr.data[:] = 0.0
        self.xc.kernel.calculate(e_r.data, nt_sr.data, vxct_sr.data,
                                 sigma_xr, dedsigma_xr,
                                 taut_sr.data, dedtaut_sr.data)
        add_gradient_correction(self.xc.grad_v, gradn_svr, sigma_xr,
                                dedsigma_xr, vxct_sr.data)
        return e_r.integrate(), vxct_sr, dedtaut_sr


class KEDCalculator:
    add: None | Callable[[Array1D, PWArray, UGArray], None] = None

    def _initialize(self, wfs: PWFDWaveFunctions):
        if self.add is None:
            if hasattr(wfs.psit_nX.desc, 'ecut'):
                self.add = pw_add_ked
            else:
                self.add = FDKEDCalculator()

    def add_ked(self,
                taut_sR,
                wfs: PWFDWaveFunctions):
        self._initialize(wfs)
        occ_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n
        self.add(occ_n, wfs.psit_nX, taut_sR[wfs.spin])


def pw_add_ked(occ_n: Array1D, psit_nG: PWArray, taut_R: UGArray) -> None:
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
            add_to_density(0.5 * f, dpsit1_R.data, taut1_R.data)
    domain_comm.sum(taut1_R.data)
    tmp_R = taut_R.new()
    tmp_R.scatter_from(taut1_R)
    taut_R.data += tmp_R.data


class FDKEDCalculator:
    def __init__(self):
        self.grad_v = []

    def __call__(self, occ_n, psit_nX, taut_R):
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
                add_to_density(0.5 * f, tmp_R.data, taut_R.data)
