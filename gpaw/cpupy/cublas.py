from gpaw.utilities.blas import rk
from gpaw.utilities.blas import mmm


def syrk(alpha, a, beta, c):
    print(a.shape, c.shape, 21354)
    rk(alpha, a.data, beta, c.data)


def gemm(transa, transb, a, b, c, alpha, beta):
    mmm(alpha,
        a.data,
        transa.replace('H', 'C'),
        b.data,
        transb.replace('H', 'C'),
        beta,
        c.data)
