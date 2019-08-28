# Copyright (C) 2003  CAMP
# Copyright (C) 2010  Argonne National Laboratory
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Linear Algebra PACKage (LAPACK)
"""

import numpy as np
import scipy.linalg as linalg

from gpaw import debug


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

    assert a.dtype in [float]
    w[:], a.T[:] = linalg.eigh(a,
                               lower=True,
                               overwrite_a=True,
                               check_finite=debug)
    return

    # info = _gpaw.diagonalize(a, w)
    # if info != 0:
    #    raise RuntimeError('diagonalize error: %d' % info)


def general_diagonalize(a, w, b, iu=None):
    """Diagonalize a generalized symmetric/hermitian matrix

    A * x = (lambda) * B * x,

    where `lambda` is the eigenvalue and `A` and `B` are the
    matrices corresponding to `a` and `b`, respectively.

    If `iu` is `None`:
    Uses dsygvd/zhegvd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered.

    If `iu` is not `None`:
    Uses dsygvx/zhegvx to find the eigenvalues of 1 through `iu`.
    Stores the eigenvectors in `z` and the eigenvalues in `w` in
    ascending order.
    """

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert a.dtype in [float, complex]
    assert w.dtype == float
    n = len(a)
    if n == 0:
        return
    assert a.shape == (n, n)
    assert w.shape == (n,)
    assert b.flags.contiguous
    assert b.dtype == a.dtype
    assert b.shape == a.shape

    if iu is not None:
        z = np.zeros((n, n), dtype=a.dtype)
        assert z.flags.contiguous

    w[:1] = 42

    if iu is None:
        w[:], a.T[:] = linalg.eigh(a, b,
                                   lower=True,
                                   overwrite_a=True,
                                   check_finite=debug)
        if a.dtype == complex:
            np.negative(a.imag, a.imag)
        return
        # info = _gpaw.general_diagonalize(a, w, b)
    else:
        w[:iu], a.T[:, :iu] = linalg.eigh(a, b,
                                          eigvals=(0, iu - 1),
                                          lower=True,
                                          overwrite_a=True,
                                          check_finite=debug)
        if a.dtype == complex:
            np.negative(a.imag, a.imag)
        return
        # info = _gpaw.general_diagonalize(a, w, b, z, iu)
        # a[:] = z
