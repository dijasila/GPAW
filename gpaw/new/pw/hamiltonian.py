import _gpaw
import numpy as np
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.core.plane_waves import PlaneWaveExpansions


class PWHamiltonian(Hamiltonian):
    def apply(self,
              vt_sR: UniformGridFunctions,
              psit_nG: PlaneWaveExpansions,
              out: PlaneWaveExpansions,
              spin: int):
        out_nG = out
        vt_R = vt_sR.data[spin]
        np.multiply(psit_nG.desc.ekin_G, psit_nG.data, out_nG.data)
        grid = vt_sR.desc
        if psit_nG.desc.dtype == complex:
            grid = grid.new(dtype=complex)
        f_R = grid.empty()
        for p_G, o_G in zip(psit_nG, out_nG):
            f_R = p_G.ifft(out=f_R)
            f_R.data *= vt_R
            o_G.data += f_R.fft(pw=p_G.desc).data
        return out_nG

    def create_preconditioner(self, blocksize):
        return precondition


def precondition(psit, residuals, out):
    G2 = psit.desc.ekin_G * 2
    for r, o, ekin in zip(residuals.data, out.data, psit.norm2('kinetic')):
        _gpaw.pw_precond(G2, r, ekin, o)


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

        np.multiply(psit_nsG.desc.ekin_G, psit_nsG.data, out_nsG.data)

        grid = vt_xR.desc.new(dtype=complex)

        v, x, y, z = vt_xR.data
        iy = y * 1j

        f_sR = grid.empty(2)
        g_R = grid.empty()

        for p_sG, o_sG in zip(psit_nsG, out_nsG):
            p_sG.ifft(out=f_sR)
            a, b = f_sR.data
            g_R.data = a * (v + z) + b * (x - iy)
            o_sG.data[0] += g_R.fft(pw=p_sG.desc).data
            g_R.data = a * (x + iy) + b * (v - z)
            o_sG.data[1] += g_R.fft(pw=p_sG.desc).data

        return out_nsG

    def create_preconditioner(self, blocksize):
        return spinor_precondition


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
