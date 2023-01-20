from gpaw.core import PlaneWaves
from gpaw.new.pw.hamiltonian import precondition
from gpaw.gpu import cupy as cp, T


def test_prec(xp):
    pw = PlaneWaves(cell=[4, 4, 4], ecut=10, dtype=complex)
    n = 2
    psit_nG, residual_nG, out_nG = pw.zeros((3, n), xp=xp)
    psit_nG.data[:, 0] = 1
    residual_nG.data[:] = 1
    with T():
        precondition(psit_nG, residual_nG, out_nG)

    ekin_n = ...
    with T():
        prec(ekin_n, ...)


@cp.fuse()
def prec(ekin, G2):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx)
