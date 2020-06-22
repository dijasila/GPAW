import numpy as np
from gpaw.utilities.blas import gemm, axpy, r2k, rk
from gpaw.utilities.tools import tri2full


def test_linalg_blas():
    a = np.arange(5 * 7).reshape(5, 7) + 4.
    a = a * (2 + 1.j)

    # Check gemm for transa='n'
    a2 = np.arange(7 * 5 * 1 * 3).reshape(7, 5, 1, 3) * (-1. + 4.j) + 3.
    c = np.tensordot(a, a2, [1, 0])
    gemm(1., a2, a, -1., c, 'n')
    assert not c.any()

    # Check gemm for transa='c'
    a = np.arange(4 * 5 * 1 * 3).reshape(4, 5, 1, 3) * (3. - 2.j) + 4.
    c = np.tensordot(a, a2.conj(), [[1, 2, 3], [1, 2, 3]])
    gemm(1., a2, a, -1., c, 'c')
    assert not c.any()

    # Check axpy
    c = 5.j * a
    axpy(-5.j, a, c)
    assert not c.any()

    # Check rk
    c = np.tensordot(a, a.conj(), [[1, 2, 3], [1, 2, 3]])
    rk(1., a, -1., c)
    tri2full(c)
    assert not c.any()

    a2.shape = 3, 7, 5, 1

    # Check r2k
    a2 = 5. * a
    c = np.tensordot(a, a2.conj(), [[1, 2, 3], [1, 2, 3]])
    r2k(.5, a, a2, -1., c)
    tri2full(c)
    assert not c.any()
