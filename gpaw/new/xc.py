from __future__ import annotations

from typing import Callable

import numpy as np

from gpaw.core import UGArray, UGDesc
from gpaw.fd_operators import Gradient
from gpaw.gpu import as_np, as_xp
from gpaw.new import zips
from gpaw.new.c import (add_to_density, add_to_density_gpu, evaluate_lda_gpu,
                        evaluate_pbe_gpu)
from gpaw.new.calculation import DFTState
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D, Array4D
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional as OldXCFunctional
from gpaw.xc.gga import add_gradient_correction, gga_vars
from gpaw.xc.mgga import MGGA


def create_functional(xc: OldXCFunctional | str | dict,
                      grid: UGDesc) -> Functional:
    if isinstance(xc, (str, dict)):
        xc = XC(xc)
    if xc.type == 'LDA':
        return LDAFunctional(xc, grid)
    if xc.type == 'GGA':
        return GGAFunctional(xc, grid)
    if xc.type == 'MGGA':
        return MGGAFunctional(xc, grid)
    raise ValueError(f'{xc.type} not supported')


class Functional:
    def __init__(self,
                 xc: OldXCFunctional,
                 grid: UGDesc):
        self.xc = xc
        self.grid = grid
        self.setup_name = self.xc.get_setup_name()
        self.name = self.xc.name
        self.type = self.xc.type
        self.xc.gd = grid._gd

    def __str__(self):
        return f'name: {self.xc.get_description()}'

    def calculate_paw_correction(self, setup, d, h=None):
        return self.xc.calculate_paw_correction(setup, d, h)

    def get_setup_name(self) -> str:
        return self.name

    def stress_contribution(self,
                            state: DFTState,
                            interpolate: Callable[[UGArray], UGArray]
                            ) -> Array2D:
        raise NotImplementedError


class LDAFunctional(Functional):
    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        xp = nt_sr.xp
        vxct_sr = nt_sr.new()
        vxct_sr.data[:] = 0.0
        e_r = nt_sr.desc.empty(xp=xp)
        if xp is np:
            self.xc.kernel.calculate(e_r.data, nt_sr.data, vxct_sr.data)
        else:
            if self.name != 'LDA':
                raise ValueError(f'{self.name} not supported on GPU')
            evaluate_lda_gpu(nt_sr.data, vxct_sr.data, e_r.data)
        exc = e_r.integrate()
        return exc, vxct_sr, None

    def stress_contribution(self,
                            state: DFTState,
                            interpolate: Callable[[UGArray], UGArray]
                            ) -> Array2D:
        ibzwfs = state.ibzwfs
        if ibzwfs.kpt_comm.rank == 0 and ibzwfs.band_comm.rank == 0:
            nt_sr = interpolate(state.density.nt_sR)
            s_vv = self.xc.stress_tensor_contribution(as_np(nt_sr.data),
                                                      skip_sum=True)
            return as_xp(s_vv, nt_sr.xp)
        return state.density.nt_sR.xp.zeros((3, 3))


class GGAFunctional(LDAFunctional):
    def __init__(self,
                 xc: OldXCFunctional,
                 grid: UGDesc,
                 stencil_range: int = 2):
        super().__init__(xc, grid)
        self.grad_v = [Gradient(grid._gd, v, n=stencil_range)
                       for v in range(3)]

    def calculate(self,
                  nt_sr: UGArray,
                  taut_sr: UGArray | None = None) -> tuple[float,
                                                           UGArray,
                                                           UGArray | None]:
        gradn_svr, sigma_xr = gradient_and_sigma(self.grad_v, nt_sr)

        xp = nt_sr.xp
        vxct_sr = nt_sr.new()
        vxct_sr.data[:] = 0.0
        dedsigma_xr = sigma_xr.new()
        e_r = nt_sr.desc.empty(xp=xp)

        if xp is np:
            self.xc.kernel.calculate(e_r.data, nt_sr.data, vxct_sr.data,
                                     sigma_xr.data, dedsigma_xr.data)
        else:
            if self.name != 'PBE':
                raise ValueError(f'{self.name} not supported on GPU')
            evaluate_pbe_gpu(nt_sr.data, vxct_sr.data, e_r.data)

        add_gradient_correction([grad.apply for grad in self.grad_v],
                                gradn_svr.data, sigma_xr.data,
                                dedsigma_xr.data, vxct_sr.data)
        exc = e_r.integrate()
        return exc, vxct_sr, None


def gradient_and_sigma(grad_v, n_sr: UGArray) -> tuple[UGArray, UGArray]:
    """Calculate gradient of density and sigma.

    Returns:::

      _    _
      ∇ n (r)
         s

    and:::

         _     _   2     _    _          _   2
      σ (r) = |∇n |, σ = ∇n . ∇n ,  σ = |∇n |
       0         0    1    0    1-   2     1
    """
    nspins = n_sr.dims[0]
    xp = n_sr.xp

    gradn_svr = n_sr.desc.empty((nspins, 3), xp=xp)
    for v, grad in enumerate(grad_v):
        for s in range(nspins):
            grad(n_sr[s], gradn_svr[s, v])

    sigma_xr = n_sr.desc.empty(nspins * 2 - 1, xp=xp)
    dot_product(gradn_svr[0], None, sigma_xr[0])
    if nspins == 2:
        dot_product(gradn_svr[0], gradn_svr[1], sigma_xr[1])
        dot_product(gradn_svr[1], None, sigma_xr[2])

    return gradn_svr, sigma_xr


def dot_product(a_vr, b_vr, out_r):
    xp = a_vr.xp
    if b_vr is None:
        out_r.data[:] = 0.0
        if xp is np:
            for a_r in a_vr.data:
                add_to_density(1.0, a_r, out_r.data)
        else:
            add_to_density_gpu(np.ones(3), a_vr.data, out_r.data)
    else:
        if xp is np:
            np.einsum('vr, vr -> r', a_vr.data, b_vr.data, out=out_r.data)
        else:
            out_r.data[:] = xp.einsum('vr, vr -> r', a_vr.data, b_vr.data)


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

    def stress_contribution(self,
                            state: DFTState,
                            interpolate: Callable[[UGArray], UGArray]
                            ) -> Array2D:
        stress_vv = np.zeros((3, 3))

        taut_swR = _taut(state.ibzwfs, state.density.nt_sR.desc)
        if taut_swR is None:
            return stress_vv

        dedtaut_sR = state.potential.dedtaut_sR
        assert dedtaut_sR is not None
        dedtaut_sr = interpolate(dedtaut_sR)
        for taut_wR, dedtaut_r in zips(taut_swR, dedtaut_sr):
            w = 0
            for v1 in range(3):
                for v2 in range(v1, 3):
                    taut_r = interpolate(taut_wR[w])
                    s = taut_r.integrate(dedtaut_r, skip_sum=True)
                    stress_vv[v1, v2] -= s
                    if v2 != v1:
                        stress_vv[v2, v1] -= s
                    w += 1

        nt_sr = interpolate(state.density.nt_sR)
        taut_sR = state.density.taut_sR
        assert taut_sR is not None
        taut_sr = interpolate(taut_sR)
        assert isinstance(self.xc, MGGA)
        stress_vv += _mgga(self.xc, nt_sr.data, taut_sr.data)

        return stress_vv


def _taut(ibzwfs: IBZWaveFunctions, grid: UGDesc) -> UGArray | None:
    """Calculate upper half of symmetric ked tensor.

    Returns ``taut_swR=taut_svvR``.  Mapping from ``w`` to ``vv``::

        0 1 2
        . 3 4
        . . 5

    Only cores with ``kpt_comm.rank==0`` and ``band_comm.rank==0``
    return the uniform grids - other cores return None.
    """
    # "1" refers to undistributed arrays
    dpsit1_vR = grid.new(comm=None, dtype=ibzwfs.dtype).empty(3)
    taut1_swR = grid.new(comm=None).zeros((ibzwfs.nspins, 6))
    assert isinstance(taut1_swR, UGArray)  # Argggghhh!
    domain_comm = grid.comm

    for wfs in ibzwfs:
        assert isinstance(wfs, PWFDWaveFunctions)
        psit_nG = wfs.psit_nX
        pw = psit_nG.desc

        pw1 = pw.new(comm=None)
        psit1_G = pw1.empty()
        iGpsit1_G = pw1.empty()
        Gplusk1_Gv = pw1.reciprocal_vectors()

        occ_n = wfs.weight * wfs.spin_degeneracy * wfs.myocc_n

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
            for Gplusk1_G, dpsit1_R in zips(Gplusk1_Gv.T, dpsit1_vR):
                iGpsit1_G.data[:] = psit1_G.data
                iGpsit1_G.data *= 1j * Gplusk1_G
                iGpsit1_G.ifft(out=dpsit1_R)
            w = 0
            for v1 in range(3):
                for v2 in range(v1, 3):
                    taut1_swR[wfs.spin, w].data += (
                        f * (dpsit1_vR[v1].data.conj() *
                             dpsit1_vR[v2].data).real)
                    w += 1

    ibzwfs.kpt_comm.sum(taut1_swR.data, 0)
    if ibzwfs.kpt_comm.rank == 0:
        ibzwfs.band_comm.sum(taut1_swR.data, 0)
        if ibzwfs.band_comm.rank == 0:
            domain_comm.sum(taut1_swR.data, 0)
            if domain_comm.rank == 0:
                symmetries = ibzwfs.ibz.symmetries
                taut1_swR.symmetrize(symmetries.rotation_scc,
                                     symmetries.translation_sc)
            taut_swR = grid.empty((ibzwfs.nspins, 6))
            taut_swR.scatter_from(taut1_swR)
            return taut_swR
    return None


def _mgga(xc: MGGA, nt_sr: Array4D, taut_sr: Array4D) -> Array2D:
    # The GGA and LDA part of this should be factored out!
    sigma_xr, dedsigma_xr, gradn_svr = gga_vars(xc.gd,
                                                xc.grad_v,
                                                nt_sr)
    nspins = len(nt_sr)
    dedtaut_sr = np.empty_like(nt_sr)
    vt_sr = xc.gd.zeros(nspins)
    e_r = xc.gd.empty()
    xc.kernel.calculate(e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr,
                        taut_sr, dedtaut_sr)

    def integrate(a1_r, a2_r=None):
        return xc.gd.integrate(a1_r, a2_r, global_integral=False)

    P = integrate(e_r)
    for vt_r, nt_r in zip(vt_sr, nt_sr):
        P -= integrate(vt_r, nt_r)
    for sigma_r, dedsigma_r in zip(sigma_xr, dedsigma_xr):
        P -= 2 * integrate(sigma_r, dedsigma_r)
    for taut_r, dedtaut_r in zip(taut_sr, dedtaut_sr):
        P -= integrate(taut_r, dedtaut_r)

    stress_vv = P * np.eye(3)
    for v1 in range(3):
        for v2 in range(3):
            stress_vv[v1, v2] -= integrate(gradn_svr[0, v1] *
                                           gradn_svr[0, v2],
                                           dedsigma_xr[0]) * 2
            if nspins == 2:
                stress_vv[v1, v2] -= integrate(gradn_svr[0, v1] *
                                               gradn_svr[1, v2],
                                               dedsigma_xr[1]) * 2
                stress_vv[v1, v2] -= integrate(gradn_svr[1, v1] *
                                               gradn_svr[1, v2],
                                               dedsigma_xr[2]) * 2
    return stress_vv
