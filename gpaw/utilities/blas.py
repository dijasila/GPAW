# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Basic Linear Algebra Subroutines (BLAS)

See also:
http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
and
http://www.netlib.org/lapack/lug/node145.html
"""
from typing import TypeVar

import numpy as np
import scipy.linalg.blas as blas

from gpaw import debug
import _gpaw

__all__ = ['mmm']

T = TypeVar('T', float, complex)


def mmm(alpha: T,
        a: np.ndarray,
        opa: str,
        b: np.ndarray,
        opb: str,
        beta: T,
        c: np.ndarray) -> None:
    """Matrix-matrix multiplication using dgemm or zgemm.

    For opa='n' and opb='n', we have::

        c <- alpha * a * b + beta * c.

    Use 't' to transpose matrices and 'c' to transpose and complex conjugate
    matrices.
    """

    assert opa in 'NTC'
    assert opb in 'NTC'

    if opa == 'N':
        a1, a2 = a.shape
    else:
        a2, a1 = a.shape
    if opb == 'N':
        b1, b2 = b.shape
    else:
        b2, b1 = b.shape
    assert a2 == b1
    assert c.shape == (a1, b2)

    assert a.strides[1] == b.strides[1] == c.strides[1] == c.itemsize
    assert a.dtype == b.dtype == c.dtype
    if a.dtype == float:
        assert not isinstance(alpha, complex)
        assert not isinstance(beta, complex)
    else:
        assert a.dtype == complex

    _gpaw.mmm(alpha, a, opa, b, opb, beta, c)


def gemm(alpha, a, b, beta, c, transa='n'):
    """General Matrix Multiply.

    Performs the operation::

      c <- alpha * b.a + beta * c

    If transa is "n", ``b.a`` denotes the matrix multiplication defined by::

                      _
                     \
      (b.a)        =  ) b  * a
           ijkl...   /_  ip   pjkl...
                      p

    If transa is "t" or "c", ``b.a`` denotes the matrix multiplication
    defined by::

                      _
                     \
      (b.a)        =  ) b    *    a
           ij        /_  iklm...   jklm...
                     klm...

    where in case of "c" also complex conjugate of a is taken.
    """
    assert np.isfinite(c).all()

    assert (a.dtype == float and b.dtype == float and c.dtype == float and
            isinstance(alpha, float) and isinstance(beta, float) or
            a.dtype == complex and b.dtype == complex and c.dtype == complex)
    if transa == 'n':
        assert a.size == 0 or a[0].flags.contiguous
        assert c.flags.contiguous or c.ndim == 2 and c.strides[1] == c.itemsize
        assert b.ndim == 2
        assert b.size == 0 or b.strides[1] == b.itemsize
        assert a.shape[0] == b.shape[1]
        assert c.shape == b.shape[0:1] + a.shape[1:]
    else:
        assert a.flags.contiguous
        assert b.size == 0 or b[0].flags.contiguous
        assert c.strides[1] == c.itemsize
        assert a.shape[1:] == b.shape[1:]
        assert c.shape == (b.shape[0], a.shape[0])
    _gpaw.gemm(alpha, a, b, beta, c, transa)


def axpy(alpha, x, y):
    """alpha x plus y.

    Performs the operation::

      y <- alpha * x + y

    """
    if x.size == 0:
        return
    x = x.ravel()
    y = y.ravel()
    if x.dtype == float:
        z = blas.daxpy(x, y, a=alpha)
    else:
        z = blas.zaxpy(x, y, a=alpha)
    assert z is y, (x, y, x.shape, y.shape)


def rk(alpha, a, beta, c, trans='c'):
    """Rank-k update of a matrix.

    Performs the operation::

                        dag
      c <- alpha * a . a    + beta * c

    where ``a.b`` denotes the matrix multiplication defined by::

                 _
                \
      (a.b)   =  ) a         * b
           ij   /_  ipklm...     pjklm...
               pklm...

    ``dag`` denotes the hermitian conjugate (complex conjugation plus a
    swap of axis 0 and 1).

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    assert beta == 0.0 or np.isfinite(c).all()

    assert (a.dtype == float and c.dtype == float or
            a.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous
    assert a.ndim > 1
    if trans == 'n':
        assert c.shape == (a.shape[1], a.shape[1])
    else:
        assert c.shape == (a.shape[0], a.shape[0])
    assert c.strides[1] == c.itemsize
    _gpaw.rk(alpha, a, beta, c, trans)


def r2k(alpha, a, b, beta, c, trans='c'):
    """Rank-2k update of a matrix.

    Performs the operation::

                        dag        cc       dag
      c <- alpha * a . b    + alpha  * b . a    + beta * c

    or if trans='n'::
                    dag           cc   dag
      c <- alpha * a   . b + alpha  * b   . a + beta * c

    where ``a.b`` denotes the matrix multiplication defined by::

                 _
                \
      (a.b)   =  ) a         * b
           ij   /_  ipklm...     pjklm...
               pklm...

    ``cc`` denotes complex conjugation.

    ``dag`` denotes the hermitian conjugate (complex conjugation plus a
    swap of axis 0 and 1).

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    assert beta == 0.0 or np.isfinite(np.tril(c)).all()

    assert (a.dtype == float and b.dtype == float and c.dtype == float or
            a.dtype == complex and b.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous and b.flags.contiguous
    assert a.ndim > 1
    assert a.shape == b.shape
    if trans == 'c':
        assert c.shape == (a.shape[0], a.shape[0])
    else:
        assert c.shape == (a.shape[1], a.shape[1])
    assert c.strides[1] == c.itemsize
    _gpaw.r2k(alpha, a, b, beta, c, trans)


def _gemmdot(a, b, alpha=1.0, beta=1.0, out=None, trans='n'):
    """Matrix multiplication using gemm.

    return reference to out, where::

      out <- alpha * a . b + beta * out

    If out is None, a suitably sized zero array will be created.

    ``a.b`` denotes matrix multiplication, where the product-sum is
    over the last dimension of a, and either
    the first dimension of b (for trans='n'), or
    the last dimension of b (for trans='t' or 'c').

    If trans='c', the complex conjugate of b is used.
    """
    # Store original shapes
    ashape = a.shape
    bshape = b.shape

    # Vector-vector multiplication is handled by dotu
    if a.ndim == 1 and b.ndim == 1:
        assert out is None
        if trans == 'c':
            return alpha * np.vdot(b, a)  # dotc conjugates *first* argument
        else:
            return alpha * a.dot(b)

    # Map all arrays to 2D arrays
    a = a.reshape(-1, a.shape[-1])
    if trans == 'n':
        b = b.reshape(b.shape[0], -1)
        outshape = a.shape[0], b.shape[1]
    else:  # 't' or 'c'
        b = b.reshape(-1, b.shape[-1])

    # Apply BLAS gemm routine
    outshape = a.shape[0], b.shape[trans == 'n']
    if out is None:
        # (ATLAS can't handle uninitialized output array)
        out = np.zeros(outshape, a.dtype)
    else:
        out = out.reshape(outshape)
    gemm(alpha, b, a, beta, out, trans)

    # Determine actual shape of result array
    if trans == 'n':
        outshape = ashape[:-1] + bshape[1:]
    else:  # 't' or 'c'
        outshape = ashape[:-1] + bshape[:-1]
    return out.reshape(outshape)


if not hasattr(_gpaw, 'mmm'):
    def gemm(alpha, a, b, beta, c, transa='n'):  # noqa
        if c.size == 0:
            return
        if beta == 0:
            c[:] = 0.0
        else:
            c *= beta
        if transa == 'n':
            c += alpha * b.dot(a.reshape((len(a), -1))).reshape(c.shape)
        elif transa == 't':
            c += alpha * b.reshape((len(b), -1)).dot(
                a.reshape((len(a), -1)).T)
        else:
            c += alpha * b.reshape((len(b), -1)).dot(
                a.reshape((len(a), -1)).T.conj())

    def rk(alpha, a, beta, c, trans='c'):  # noqa
        if c.size == 0:
            return
        if beta == 0:
            c[:] = 0.0
        else:
            c *= beta
        if trans == 'n':
            c += alpha * a.conj().T.dot(a)
        else:
            a = a.reshape((len(a), -1))
            c += alpha * a.dot(a.conj().T)

    def r2k(alpha, a, b, beta, c, trans='c'):  # noqa
        if c.size == 0:
            return
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        if trans == 'c':
            c += (alpha * a.reshape((len(a), -1))
                  .dot(b.reshape((len(b), -1)).conj().T) +
                  alpha * b.reshape((len(b), -1))
                  .dot(a.reshape((len(a), -1)).conj().T))
        else:
            c += alpha * (a.conj().T @ b + b.conj().T @ a)

    def op(o, m):
        if o == 'N':
            return m
        if o == 'T':
            return m.T
        return m.conj().T

    def mmm(alpha: T, a: np.ndarray, opa: str,  # noqa
            b: np.ndarray, opb: str,
            beta: T, c: np.ndarray) -> None:
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        c += alpha * op(opa, a).dot(op(opb, b))

    gemmdot = _gemmdot

elif not debug:
    mmm = _gpaw.mmm  # noqa
    gemm = _gpaw.gemm  # noqa
    rk = _gpaw.rk  # noqa
    r2k = _gpaw.r2k  # noqa
    gemmdot = _gemmdot

else:
    def gemmdot(a, b, alpha=1.0, beta=1.0, out=None, trans='n'):
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert a.dtype == b.dtype
        if trans == 'n':
            assert a.shape[-1] == b.shape[0]
        else:
            assert a.shape[-1] == b.shape[-1]
        if out is not None:
            assert out.flags.contiguous
            assert a.dtype == out.dtype
            assert a.ndim > 1 or b.ndim > 1
            if trans == 'n':
                assert out.shape == a.shape[:-1] + b.shape[1:]
            else:
                assert out.shape == a.shape[:-1] + b.shape[:-1]
        return _gemmdot(a, b, alpha, beta, out, trans)
