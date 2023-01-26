import _gpaw
import numpy as np
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.core.plane_waves import PlaneWaveExpansions
from gpaw.gpu import cupy as cp


class PWHamiltonian(Hamiltonian):
    def apply(self,
              vt_sR: UniformGridFunctions,
              psit_nG: PlaneWaveExpansions,
              out: PlaneWaveExpansions,
              spin: int):
        out_nG = out
        vt_R = vt_sR.data[spin]
        xp = psit_nG.xp
        e_kin_G = xp.asarray(psit_nG.desc.ekin_G)
        xp.multiply(e_kin_G, psit_nG.data, out_nG.data)
        grid = vt_sR.desc
        if psit_nG.desc.dtype == complex:
            grid = grid.new(dtype=complex)
        f_R = grid.empty(xp=xp)
        for p_G, o_G in zip(psit_nG, out_nG):
            f_R = p_G.ifft(out=f_R)
            f_R.data *= vt_R
            o_G.data += f_R.fft(pw=p_G.desc).data
        return out_nG

    def create_preconditioner(self, blocksize):
        return precondition


# Old optimized code:
"""
        N = len(psit_xG)
        S = self.gd.comm.size

        vt_R = self.gd.collect(ham.vt_sG[kpt.s], broadcast=True)
        Q_G = self.pd.Q_qG[kpt.q]
        T_G = 0.5 * self.pd.G2_qG[kpt.q]

        for n1 in range(0, N, S):
            n2 = min(n1 + S, N)
            psit_G = self.pd.alltoall1(psit_xG[n1:n2], kpt.q)
            with self.timer('HMM T'):
                np.multiply(T_G, psit_xG[n1:n2], Htpsit_xG[n1:n2])
            if psit_G is not None:
                psit_R = self.pd.ifft(psit_G, kpt.q, local=True, safe=False)
                psit_R *= vt_R
                self.pd.fftplan.execute()
                vtpsit_G = self.pd.tmp_Q.ravel()[Q_G]
            else:
                vtpsit_G = self.pd.tmp_G
            self.pd.alltoall2(vtpsit_G, kpt.q, Htpsit_xG[n1:n2])

        ham.xc.apply_orbital_dependent_hamiltonian(
            kpt, psit_xG, Htpsit_xG, ham.dH_asp)
"""


def precondition(psit_nG, residual_nG, out):
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
        for r_G, o_G, ekin in zip(residual_nG.data,
                                  out.data,
                                  ekin_n):
            _gpaw.pw_precond(G2_G, r_G, ekin, o_G)
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
    for r_sG, o_sG, ekin in zip(residual_nsG.data,
                                out.data,
                                psit_nsG.norm2('kinetic').sum(1)):
        for r_G, o_G in zip(r_sG, o_sG):
            _gpaw.pw_precond(G2_G, r_G, ekin, o_G)


class SpinorPWHamiltonian(Hamiltonian):
    def apply(self,
              vt_xR: UniformGridFunctions,
              psit_nsG: PlaneWaveExpansions,
              out: PlaneWaveExpansions,
              spin: int):
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

        for p_sG, o_sG in zip(psit_nsG, out_nsG):
            p_sG.ifft(out=f_sR)
            a, b = f_sR.data
            g_R.data = a * (v + z) + b * (x - iy)
            o_sG.data[0] += g_R.fft(pw=pw).data
            g_R.data = a * (x + iy) + b * (v - z)
            o_sG.data[1] += g_R.fft(pw=pw).data

        return out_nsG

    def create_preconditioner(self, blocksize):
        return spinor_precondition
