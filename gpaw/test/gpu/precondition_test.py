import numpy as np
from gpaw.core import PlaneWaves
from gpaw.gpu import T
from gpaw.gpu import cupy as cp
from gpaw.new.pw.hamiltonian import precondition


def test_prec(xp):
    a = 6
    pw = PlaneWaves(cell=[a, a, a], ecut=800 / 27, dtype=complex)
    n = 50
    psit_nG, residual_nG, out_nG = pw.zeros((3, n), xp=xp)
    psit_nG.data[:, :2] = 1.0
    residual_nG.data[:] = 1.0
    print(residual_nG.data[:].shape)
    for _ in range(5):
        with T():
            precondition(psit_nG, residual_nG, out_nG)

        ekin_n = psit_nG.norm2('kinetic')
        G2_G = xp.asarray(psit_nG.desc.ekin_G * 2)
        with T():
            out_nG.data[:] = prec(ekin_n[:, np.newaxis],
                                  G2_G[np.newaxis],
                                  psit_nG.data)
        if xp == np:
            continue

        psit_nGc = psit_nG.data.view(float).reshape((n, -1, 2))
        with T():
            # out_nGc = out_nG.data.view(float).reshape((n, -1, 2))
            # out_nGc[:] =
            prec2(ekin_n[:, np.newaxis, np.newaxis],
                  G2_G[np.newaxis, :, np.newaxis],
                  psit_nGc)


@cp.fuse()
def prec(ekin, G2, psit):
    x = 1 / ekin / 3 * G2
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    return -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * psit


grr = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    """
    T d = x - y;
    z = d * d;
    """,
    'squared_diff')
x = cp.arange(10, dtype=np.float64).reshape(2, 5)
y = cp.arange(5, dtype=np.float64)
z = grr(x, y)
print(z)

prec2 = cp.ElementwiseKernel(
    'T ekin, T G2, T psit',
    'T out',
    """
    T x = 1 / ekin / 3 * G2;
    T a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0));
    T xx = x * x;
    out = -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * psit;
    """,
    'prec2')

test_prec(np)
test_prec(cp)
