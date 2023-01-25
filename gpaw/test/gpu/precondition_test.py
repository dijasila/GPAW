import numpy as np
from gpaw.core import PlaneWaves
from gpaw.gpu import T
from gpaw.gpu import cupy as cp
from gpaw.new.pw.hamiltonian import precondition


def test_prec(xp):
    pw = PlaneWaves(cell=[4, 4, 4], ecut=10, dtype=complex)
    n = 2
    psit_nG, residual_nG, out_nG = pw.zeros((3, n), xp=xp)
    psit_nG.data[:, :2] = 1.0
    residual_nG.data[:] = 1.0
    with T():
        precondition(psit_nG, residual_nG, out_nG)

    ekin_n = psit_nG.norm2('kinetic')
    G2_G = xp.asarray(psit_nG.desc.ekin_G * 2)
    with T():
        prec(ekin_n[:, np.newaxis], G2_G[np.newaxis], psit_nG.data)


@cp.fuse()
def prec(ekin, G2, psit):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * psit


test_prec(np)
test_prec(cp)
