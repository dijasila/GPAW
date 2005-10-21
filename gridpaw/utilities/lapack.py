# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw import _gridpaw
from gridpaw import debug


def diagonalize(a, w, b=None):
    """Diagonalize a symmetric/hermitian matrix.

    Uses dsyevd/zheevd to diagonalize symmetric/hermitian matrix
    `a`.  The eigenvectors are returned in `a` and the eigenvalues
    in `w` in ascending order.

    If a symmetric/hermitian positive definite matrix b is given, then
    dsygvd/zhegvd is used to solve a generalized eigenvalue
    problem: a*v=b*v*w."""

    assert a.iscontiguous()
    assert w.iscontiguous()
    assert a.typecode() in [num.Float, num.Complex]
    assert w.typecode() == num.Float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    if b:
        assert b.iscontiguous()
        assert b.typecode() == a.typecode()
        assert b.shape == a.shape
        info = _gridpaw.diagonalize(a, w, b)
    else:
        info = _gridpaw.diagonalize(a, w)
    return info


if not debug:
    diagonalize = _gridpaw.diagonalize
