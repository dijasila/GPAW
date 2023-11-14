"""
Authors (see AUTHORS file for details): DALV

Multipole interpolation:
                        - analytical solution for 1-3 poles
                        - Linear solver for n poles*
                        - Pade-Thiele solver for n poles*

Failure condition for the position of the poles*

*DA. Leon et al, PRB 104, 115157 (2021)

Notes:

  1) X(w) is approximated as a sum of poles
  2) Form of one pole: -P/(w**2-Q) = 2*E*R/(w**2-E**2)
  3) The input are two w and X(w) for each pole
  4) The output are E and R coefficients
  5) Use real(R), imaginary(I) or complex(C) w

**The module works for scalar polarizabilities, so if
  one wants the solution for matrix element X(G,G',q)
  then RQ_solver should be called for each G, G' and q.
"""
from __future__ import annotations
from typing import Tuple, no_type_check
from gpaw.typing import Array1D, Array2D, Array3D
import numpy as np
from numpy.linalg import eigvals
from numpy.polynomial import polynomial as poly


def fit_residue(
    npr_GG: Array2D, omega_w: Array1D, X_wGG: Array3D, E_pGG: Array3D
) -> Array3D:
    npols = len(E_pGG)
    nw = len(omega_w)
    A_GGwp = np.zeros((*E_pGG.shape[1:], nw, npols), dtype=np.complex128)
    b_GGw = np.zeros((*E_pGG.shape[1:], nw), dtype=np.complex128)
    for w in range(nw):
        A_GGwp[:, :, w, :] = (
            2 * E_pGG / (omega_w[w]**2 - E_pGG**2)).transpose((1, 2, 0))
        b_GGw[:, :, w] = X_wGG[w, :, :]

    for p in range(npols):
        for w in range(A_GGwp.shape[2]):
            A_GGwp[:, :, w, p][p >= npr_GG] = 0.0

    temp_GGp = np.einsum('GHwp,GHw->GHp',
                         A_GGwp.conj(), b_GGw)
    XTX_GGpp = np.einsum('GHwp,GHwo->GHpo',
                         A_GGwp.conj(), A_GGwp)

    if XTX_GGpp.shape[2] == 1:
        # 1D matrix, invert the number
        XTX_GGpp = 1 / XTX_GGpp
        R_GGp = np.einsum('GHpo,GHo->GHp',
                          XTX_GGpp, temp_GGp)
    else:
        try:
            R_GGp = np.linalg.solve(XTX_GGpp, temp_GGp)
        except np.linalg.LinAlgError:
            XTX_GGpp = np.linalg.pinv(XTX_GGpp)
            R_GGp = np.einsum('GHpo,GHo->GHp',
                              XTX_GGpp, temp_GGp)

    return R_GGp.transpose((2, 0, 1))


class Solver:
    def __init__(self, omega_w: Array1D):
        assert len(omega_w) % 2 == 0
        self.omega_w = omega_w
        self.npoles = len(omega_w) // 2
        self.threshold = 1e-5
        self.epsilon = 1e-8

    def solve(self, X_wG):
        raise NotImplementedError


class SinglePoleSolver(Solver):
    def __init__(self, omega_w: Array1D):
        Solver.__init__(self, omega_w=omega_w)

    def solve(self, X_wGG: Array3D) -> Tuple[Array2D, Array2D]:
        assert len(X_wGG) == 2

        omega_w = self.omega_w
        E_GG = ((X_wGG[0, :, :] * omega_w[0]**2 -
                 X_wGG[1, :, :] * omega_w[1]**2) /
                (X_wGG[0, :, :] - X_wGG[1, :, :])
                )

        def branch_sqrt_inplace(E_GG: Array2D):
            E_GG.real = np.abs(E_GG.real)  # physical pole
            E_GG.imag = -np.abs(E_GG.imag)  # correct time ordering
            E_GG[:] = np.emath.sqrt(E_GG)

        branch_sqrt_inplace(E_GG)
        mask = E_GG < self.threshold  # null pole
        E_GG[mask] = self.threshold - 1j * self.epsilon

        R_GG = fit_residue(
            npr_GG=np.zeros_like(E_GG) + 1,
            omega_w=omega_w,
            X_wGG=X_wGG,
            E_pGG=E_GG.reshape((1, *E_GG.shape)))[0, :, :]

        return E_GG, R_GG


class MultipoleSolver(Solver):
    def __init__(self, omega_w: Array1D):
        Solver.__init__(self, omega_w=omega_w)

    def solve(self, X_wGG: Array3D) -> Tuple[Array3D, Array3D]:
        assert len(X_wGG) == 2 * self.npoles

        E_GGp, npr_GG = Pade_solver(X_wGG, self.omega_w**2)
        E_pGG = E_GGp.transpose((2, 0, 1))
        R_pGG = fit_residue(npr_GG, self.omega_w, X_wGG, E_pGG)
        return E_pGG, R_pGG


def RESolver(omega_w: Array1D):
    assert len(omega_w) % 2 == 0
    npoles = len(omega_w) / 2
    assert npoles > 0
    if npoles == 1:
        return SinglePoleSolver(omega_w)
    else:
        return MultipoleSolver(omega_w)


def mpa_cond_vectorized(
    npols: int, z_w: Array1D, E_GGp: Array3D, pole_resolution: float = 1e-5
) -> Tuple[Array3D, Array2D]:
    wmax = np.max(np.real(np.emath.sqrt(z_w))) * 1.5

    E_GGp = np.emath.sqrt(E_GGp)
    args = np.abs(E_GGp.real), np.abs(E_GGp.imag)
    E_GGp = np.maximum(*args) - 1j * np.minimum(*args)
    E_GGp.sort(axis=2)  # Sort according to real part

    for i in range(npols):
        out_poles_GG = E_GGp[:, :, i].real > wmax
        E_GGp[out_poles_GG, i] = 2 * wmax - 0.01j
        for j in range(i + 1, npols):
            diff = E_GGp[:, :, j].real - E_GGp[:, :, i].real
            equal_poles_GG = diff < pole_resolution
            if np.sum(equal_poles_GG.ravel()):
                break

            # if the poles are to close, move them to the end, set value
            # to sort to the end of array (e.g., 2x wmax)
            E_GGp[:, :, j] = np.where(
                equal_poles_GG,
                (E_GGp[:, :, j].real + E_GGp[:, :, i].real) / 2
                + 1j * np.maximum(E_GGp[:, :, i].imag, E_GGp[:, :, j].imag),
                E_GGp[:, :, j],
            )
            E_GGp[equal_poles_GG, i] = 2 * wmax - 0.01j

    E_GGp.sort(axis=2)  # Sort according to real part

    npr_GG = np.sum(E_GGp.real < wmax, axis=2)
    return E_GGp, npr_GG


@no_type_check
def Pade_solver(X_wGG: Array3D, z_w: Array1D) -> Tuple[Array3D, Array2D]:
    nw, nG1, nG2 = X_wGG.shape
    npols = nw // 2

    b_GGm = np.zeros((nG1, nG2, npols + 1), dtype=np.complex128)
    bm1_GGm = b_GGm
    b_GGm[..., 0] = 1.0
    c_GGw = X_wGG.transpose((1, 2, 0)).copy()

    for i in range(1, 2 * npols):
        cm1_GGw = np.copy(c_GGw)
        bm2_GGm = np.copy(bm1_GGm)
        bm1_GGm = np.copy(b_GGm)

        c_GGw[..., i:] = (
            (cm1_GGw[..., i - 1, np.newaxis] - cm1_GGw[..., i:]) /
            ((z_w[i:] - z_w[i - 1]) * cm1_GGw[..., i:])
        )

        b_GGm = bm1_GGm - z_w[i - 1] * c_GGw[..., i, np.newaxis] * bm2_GGm
        bm2_GGm[..., npols:0:-1] = (
            c_GGw[..., i, np.newaxis] * bm2_GGm[..., npols - 1::-1]
        )
        b_GGm[..., 1:] += bm2_GGm[..., 1:]

    companion_GGmm = np.empty((nG1, nG2, npols, npols),
                              dtype=np.complex128)

    for i in range(nG1):
        for j in range(nG2):
            companion_GGmm[i, j] = poly.polycompanion(b_GGm[i, j, :npols + 1])

    E_GGm = eigvals(companion_GGmm)
    Esqr_GGm = E_GGm.copy()

    E2_GGm, npr2_GG = mpa_cond_vectorized(npols=npols, z_w=z_w, E_GGp=Esqr_GGm)

    return E2_GGm, npr2_GG
