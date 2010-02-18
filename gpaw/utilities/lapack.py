# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Linear Algebra PACKage (LAPACK)
"""

import numpy as np

from gpaw import debug
from gpaw.utilities import scalapack
from gpaw import sl_diagonalize, sl_inverse_cholesky
import _gpaw
from gpaw.utilities.tools import tri2full
from gpaw.utilities.blas import gemm


def diagonalize(a, w):
    """Diagonalize a symmetric/hermitian matrix.

    Uses dsyevd/zheevd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert a.dtype in [float, complex]
    assert w.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)

    info = _gpaw.diagonalize(a, w)
    return info

def sldiagonalize(a, w, blockcomm, root=0):
    """Diagonalize a symmetric/hermitian matrix.

    Uses dsyevd/zheevd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert a.dtype in [float, complex]
    assert w.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)

    assert scalapack()
    assert len(sl_diagonalize) == 4
    size = blockcomm.size
    assert sl_diagonalize[0]*sl_diagonalize[1] <= size
    # symmetrize the matrix
    tri2full(a)
    info = blockcomm.diagonalize(a, w,
                                 sl_diagonalize[0],
                                 sl_diagonalize[1],
                                 sl_diagonalize[2], root)
    return info

def general_diagonalize(a, w, b):
    """Diagonalize a generalized symmetric/hermitian matrix.

    Uses dsygvd/zhegvd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert a.dtype in [float, complex]
    assert w.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    assert b.flags.contiguous
    assert b.dtype == a.dtype
    assert b.shape == a.shape

    info = _gpaw.general_diagonalize(a, w, b)
    return info

def slgeneral_diagonalize(a, w, b, blockcomm, root=0):
    """Diagonalize a generalized symmetric/hermitian matrix.

    Uses dsygvd/zhegvd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert a.dtype in [float, complex]
    assert w.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    assert b.flags.contiguous
    assert b.dtype == a.dtype
    assert b.shape == a.shape

    assert scalapack()
    assert len(sl_diagonalize) == 4
    size = blockcomm.size
    assert sl_diagonalize[0]*sl_diagonalize[1] <= size
    # symmetrize the matrix
    tri2full(a)
    tri2full(b)
    info = blockcomm.diagonalize(a, w,
                                 sl_diagonalize[0],
                                 sl_diagonalize[1],
                                 sl_diagonalize[2], root, b)

    return info

def inverse_cholesky(a):
    """Calculate the inverse of the Cholesky decomposition of
    a symmetric/hermitian positive definite matrix `a`.

    Uses dpotrf/zpotrf to calculate the decomposition and then
    dtrtri/ztrtri for the inversion"""

    assert a.flags.contiguous
    assert a.dtype in [float, complex]
    n = len(a)
    assert a.shape == (n, n)

    info = _gpaw.inverse_cholesky(a)
    return info

def slinverse_cholesky(a, blockcomm, root=0):
    """Calculate the inverse of the Cholesky decomposition of
    a symmetric/hermitian positive definite matrix `a`.

    Uses dpotrf/zpotrf to calculate the decomposition and then
    dtrtri/ztrtri for the inversion"""

    assert a.flags.contiguous
    assert a.dtype in [float, complex]
    n = len(a)
    assert a.shape == (n, n)

    assert scalapack()

    assert len(sl_inverse_cholesky) == 4
    size = blockcomm.size
    assert sl_inverse_cholesky[0]*sl_inverse_cholesky[1] <= size
    # symmetrize the matrix
    tri2full(a)
    info = blockcomm.inverse_cholesky(a,
                                  sl_inverse_cholesky[0],
                                  sl_inverse_cholesky[1],
                                  sl_inverse_cholesky[2], root)
    return info

def inverse_general(a):
    assert a.dtype in [float, complex]
    n = len(a)
    assert a.shape == (n, n)
    info = _gpaw.inverse_general(a)
    return info 

def inverse_symmetric(a):
    assert a.dtype in [float, complex]
    n = len(a)
    assert a.shape == (n, n)
    info = _gpaw.inverse_symmetric(a)
    tri2full(a, 'L', 'symm')
    return info 

def right_eigenvectors(a, w, v):
    """Get right eigenvectors and eigenvalues from a square matrix
    using LAPACK dgeev.

    The right eigenvector corresponding to eigenvalue w[i] is v[i]."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert v.flags.contiguous
    assert a.dtype == float
    assert w.dtype == float
    assert v.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    assert w.shape == (n,n)
    return _gpaw.right_eigenvectors(a, w, v)

def pm(M):
    """print a matrix or a vector in mathematica style"""
    string=''
    s = M.shape
    if len(s) > 1:
        (n,m)=s
        string += '{'
        for i in range(n):
            string += '{'
            for j in range(m):
                string += str(M[i,j])
                if j == m-1:
                    string += '}'
                else:
                    string += ','
            if i == n-1:
                string += '}'
            else:
                string += ','
    else:
        n=s[0]
        string += '{'
        for i in range(n):
            string += str(M[i])
            if i == n-1:
                string += '}'
            else:
                string += ','
    return string

def sqrt_matrix(a, preserve=False):
    """Get the sqrt of a symmetric matrix a (diagonalize is used).
    The matrix is kept if preserve=True, a=sqrt(a) otherwise."""
    n = len(a)
    if debug:
         assert a.flags.contiguous
         assert a.dtype == float
         assert a.shape == (n, n)
    if preserve:
        b = a.copy()
    else:
        b = a

    # diagonalize to get the form b = Z * D * Z^T
    # where D is diagonal
    D = np.empty((n,))
    diagonalize(b, D)
    ZT = b.copy()
    Z = np.transpose(b)

    # c = Z * sqrt(D)
    c = Z * np.sqrt(D)

    # sqrt(b) = c * Z^T
    gemm(1., ZT, c, 0., b)

    return b

if not debug:
    # Bypass the Python wrappers
    right_eigenvectors = _gpaw.right_eigenvectors

    # For ScaLAPACK, we can't bypass the Python wrappers!
    if not sl_diagonalize:
        diagonalize = _gpaw.diagonalize
    if not sl_diagonalize:
        general_diagonalize = _gpaw.general_diagonalize
    if not sl_inverse_cholesky:
        inverse_cholesky = _gpaw.inverse_cholesky
