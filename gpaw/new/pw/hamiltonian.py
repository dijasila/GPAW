from typing import Callable

import numpy as np

from gpaw.core.plane_waves import PWArray
from gpaw.core.uniform_grid import UGArray
from gpaw.gpu import cupy as cp
from gpaw.new import zips
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.c import pw_precond


class PWHamiltonian(Hamiltonian):
    def __init__(self, grid, pw, xp):
        self.plan = grid.new(dtype=pw.dtype).fft_plans(xp=xp)
        self.pw_cache = {}

    def apply(self,
              vt_sR: UGArray,
              dedtaut_sR: UGArray | None,
              psit_nG: PWArray,
              out: PWArray,
              spin: int) -> PWArray:
        self.apply_local_potential(vt_sR[spin], psit_nG, out)
        if dedtaut_sR is not None:
            self.apply_mgga(dedtaut_sR[spin], psit_nG, out)
        return out

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nG: PWArray,
                              out: PWArray
                              ) -> None:
        out_nG = out
        vt_R = vt_R.gather(broadcast=True)
        xp = psit_nG.xp
        grid = vt_R.desc.new(comm=None, dtype=psit_nG.desc.dtype)
        tmp_R = grid.empty(xp=xp)
        pw = psit_nG.desc
        if pw.comm.size == 1:
            pw_local = pw
        else:
            key = tuple(pw.kpt_c)
            pw_local = self.pw_cache.get(key)
            if pw_local is None:
                pw_local = pw.new(comm=None)
                self.pw_cache[key] = pw_local
        psit_G = pw_local.empty(xp=xp)
        e_kin_G = xp.asarray(psit_G.desc.ekin_G)
        domain_comm = psit_nG.desc.comm
        mynbands = psit_nG.mydims[0]
        vtpsit_G = pw_local.empty(xp=xp)
        for n1 in range(0, mynbands, domain_comm.size):
            n2 = min(n1 + domain_comm.size, mynbands)
            psit_nG[n1:n2].gather_all(psit_G)
            if domain_comm.rank < n2 - n1:
                psit_G.ifft(out=tmp_R)
                tmp_R.data *= vt_R.data
                tmp_R.fft(out=vtpsit_G)
                psit_G.data *= e_kin_G
                vtpsit_G.data += psit_G.data
            out_nG[n1:n2].scatter_from_all(vtpsit_G)

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nG: PWArray,
                   vt_nG: PWArray) -> None:
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

    def create_preconditioner(self,
                              blocksize: int
                              ) -> Callable[[PWArray,
                                             PWArray,
                                             PWArray], None]:
        return precondition


def precondition(psit_nG: PWArray,
                 residual_nG: PWArray,
                 out: PWArray) -> None:
    """Preconditioner for KS equation.

    From:

      Teter, Payne and Allen, Phys. Rev. B 40, 12255 (1989)

    as modified by:

      Kresse and Furthmüller, Phys. Rev. B 54, 11169 (1996)
    """

    xp = psit_nG.xp
    G2_G = xp.asarray(psit_nG.desc.ekin_G * 2)
    ekin_n = psit_nG.norm2('kinetic')

    if xp is np:
        for r_G, o_G, ekin in zips(residual_nG.data,
                                   out.data,
                                   ekin_n):
            pw_precond(G2_G, r_G, ekin, o_G)
        return

    out.data[:] = gpu_prec(ekin_n[:, np.newaxis],
                           G2_G[np.newaxis],
                           residual_nG.data)


@cp.fuse()
def gpu_prec(ekin, G2, residual):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * residual


def spinor_precondition(psit_nsG, residual_nsG, out):
    G2_G = psit_nsG.desc.ekin_G * 2
    for r_sG, o_sG, ekin in zips(residual_nsG.data,
                                 out.data,
                                 psit_nsG.norm2('kinetic').sum(1)):
        for r_G, o_G in zips(r_sG, o_sG):
            pw_precond(G2_G, r_G, ekin, o_G)


class SpinorPWHamiltonian(Hamiltonian):
    def apply(self,
              vt_xR: UGArray,
              dedtaut_xR: UGArray | None,
              psit_nsG: PWArray,
              out: PWArray,
              spin: int):
        assert dedtaut_xR is None
        out_nsG = out
        pw = psit_nsG.desc

        if pw.qspiral_v is None:
            np.multiply(pw.ekin_G, psit_nsG.data, out_nsG.data)
        else:
            for s, sign in enumerate([1, -1]):
                ekin_G = 0.5 * ((pw.G_plus_k_Gv +
                                 0.5 * sign * pw.qspiral_v)**2).sum(1)
                np.multiply(ekin_G, psit_nsG.data[:, s], out_nsG.data[:, s])

        grid = vt_xR.desc.new(dtype=complex)

        v, x, y, z = vt_xR.data
        iy = y * 1j

        f_sR = grid.empty(2)
        g_R = grid.empty()

        for p_sG, o_sG in zips(psit_nsG, out_nsG):
            p_sG.ifft(out=f_sR)
            a, b = f_sR.data
            g_R.data = a * (v + z) + b * (x - iy)
            o_sG.data[0] += g_R.fft(pw=pw).data
            g_R.data = a * (x + iy) + b * (v - z)
            o_sG.data[1] += g_R.fft(pw=pw).data

        return out_nsG

    def create_preconditioner(self, blocksize):
        return spinor_precondition
