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
from gpaw import gpu
import _gpaw


__all__ = ['mmm']

T = TypeVar('T', float, complex)


def mmm(alpha: T,
        a: np.ndarray,
        opa: str,
        b: np.ndarray,
        opb: str,
        beta: T,
        c: np.ndarray,
        cuda: bool = None) -> None:
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

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    b_cpu, b_gpu = (None, b) if gpu.is_device_array(b) \
                             else (b, None)
    c_cpu, c_gpu = (None, c) if gpu.is_device_array(c) \
                             else (c, None)

    if cuda or (cuda is None and gpu.is_device_array(c)):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if b_gpu is None:
            b_gpu = gpu.copy_to_device(b_cpu)
        if c_gpu is None:
            c_gpu = gpu.copy_to_device(c_cpu)
        m = b2
        n = a1
        k = b1
        lda = a_gpu.strides[0] // a_gpu.itemsize
        ldb = b_gpu.strides[0] // b_gpu.itemsize
        ldc = c_gpu.strides[0] // c_gpu.itemsize
        _gpaw.mmm_gpu(alpha, gpu.array.get_pointer(a_gpu), lda, opa,
                      gpu.array.get_pointer(b_gpu), ldb, opb, beta,
                      gpu.array.get_pointer(c_gpu), ldc, c_gpu.itemsize,
                      m, n, k)
        if c_cpu is not None:
            gpu.copy_to_host(c_gpu, c_cpu)
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if b_cpu is None:
            b_cpu = gpu.copy_to_host(b_gpu)
        if c_cpu is None:
            c_cpu = gpu.copy_to_host(c_gpu)
        _gpaw.mmm(alpha, a_cpu, opa, b_cpu, opb, beta, c_cpu)
        if c_gpu is not None:
            gpu.copy_to_device(c_cpu, c_gpu)


def scal(alpha, x):
    """alpha x

    Performs the operation::

      x <- alpha * x

    """
    if debug:
        if isinstance(alpha, complex):
            assert is_contiguous(x, complex)
        else:
            assert isinstance(alpha, float)
            assert x.dtype in [float, complex]
            assert x.flags.contiguous

    if gpu.is_device_array(x):
        if gpu.debug:
            x_cpu = gpu.copy_to_host(x)
            _gpaw.scal(alpha, x_cpu)
        _gpaw.scal_cuda_gpu(alpha, gpu.array.get_pointer(x), x.shape, x.dtype)
        if gpu.debug:
            gpu.debug_test(x_cpu, x, "scal")
    else:
        _gpaw.scal(alpha, x)


def gemm(alpha, a, b, beta, c, transa='n', cuda=False, hybrid=False):
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
    if debug:
        assert np.isfinite(c).all()

        assert (a.dtype == float and b.dtype == float and c.dtype == float and
                isinstance(alpha, float) and isinstance(beta, float) or
                a.dtype == complex and b.dtype == complex and
                c.dtype == complex)
        assert a.flags.contiguous
        if transa == 'n':
            assert c.flags.contiguous or c.ndim == 2 and c.strides[1] == c.itemsize
            assert b.ndim == 2
            assert b.strides[1] == b.itemsize
            assert a.shape[0] == b.shape[1]
            assert c.shape == b.shape[0:1] + a.shape[1:]
        else:
            assert b.size == 0 or b[0].flags.contiguous
            assert c.strides[1] == c.itemsize
            assert a.shape[1:] == b.shape[1:]
            assert c.shape == (b.shape[0], a.shape[0])

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    b_cpu, b_gpu = (None, b) if gpu.is_device_array(b) \
                             else (b, None)
    c_cpu, c_gpu = (None, c) if gpu.is_device_array(c) \
                             else (c, None)

    if cuda or (cuda is None and gpu.is_device_array(c)):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if b_gpu is None:
            b_gpu = gpu.copy_to_device(b_cpu)
        if c_gpu is None:
            c_gpu = gpu.copy_to_device(c_cpu)
        if gpu.debug:
            a_debug = gpu.copy_to_host(a_gpu)
            b_debug = gpu.copy_to_host(b_gpu)
            c_debug = gpu.copy_to_host(c_gpu)
            _gpaw.gemm(alpha, a_debug, b_debug, beta, c_debug, transa)
        _gpaw.gemm_cuda_gpu(alpha, gpu.array.get_pointer(a_gpu), a_gpu.shape,
                            gpu.array.get_pointer(b_gpu), b_gpu.shape, beta,
                            gpu.array.get_pointer(c_gpu), c_gpu.shape,
                            a_gpu.dtype, transa, hybrid)
        if gpu.debug:
            gpu.debug_test(c_debug, c_gpu, "gemm")
        if c_cpu is not None:
            gpu.copy_to_host(c_gpu, c_cpu)
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if b_cpu is None:
            b_cpu = gpu.copy_to_host(b_gpu)
        if c_cpu is None:
            c_cpu = gpu.copy_to_host(c_gpu)
        _gpaw.gemm(alpha, a_cpu, b_cpu, beta, c_cpu, transa)
        if c_gpu is not None:
            gpu.copy_to_device(c_cpu, c_gpu)


def gemv(alpha, a, x, beta, y, trans='t', cuda=False):
    """General Matrix Vector product.

    Performs the operation::

      y <- alpha * a.x + beta * y

    ``a.x`` denotes matrix multiplication, where the product-sum is
    over the entire length of the vector x and
    the first dimension of a (for trans='n'), or
    the last dimension of a (for trans='t' or 'c').

    If trans='c', the complex conjugate of a is used. The default is
    trans='t', i.e. behaviour like np.dot with a 2D matrix and a vector.

    Example::

      >>> y_m = np.dot(A_mn, x_n)
      >>> # or better yet
      >>> y_m = np.zeros(A_mn.shape[0], A_mn.dtype)
      >>> gemv(1.0, A_mn, x_n, 0.0, y_m)

    """
    if debug:
        assert (a.dtype == float and x.dtype == float and y.dtype == float and
                isinstance(alpha, float) and isinstance(beta, float) or
                a.dtype == complex and x.dtype == complex and y.dtype == complex)
        assert a.flags.contiguous
        assert y.flags.contiguous
        assert x.ndim == 1
        assert y.ndim == a.ndim - 1
        if trans == 'n':
            assert a.shape[0] == x.shape[0]
            assert a.shape[1:] == y.shape
        else:
            assert a.shape[-1] == x.shape[0]
            assert a.shape[:-1] == y.shape

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    x_cpu, x_gpu = (None, x) if gpu.is_device_array(x) \
                             else (x, None)
    y_cpu, y_gpu = (None, y) if gpu.is_device_array(y) \
                             else (y, None)

    if cuda or (cuda is None and gpu.is_device_array(y)):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if x_gpu is None:
            x_gpu = gpu.copy_to_device(x_cpu)
        if y_gpu is None:
            y_gpu = gpu.copy_to_device(y_cpu)
        if gpu.debug:
            a_debug = gpu.copy_to_host(a_gpu)
            x_debug = gpu.copy_to_host(x_gpu)
            y_debug = gpu.copy_to_host(y_gpu)
            _gpaw.gemv(alpha, a_debug, x_debug, beta, y_debug, trans)
        _gpaw.gemv_cuda_gpu(alpha, gpu.array.get_pointer(a_gpu), a_gpu.shape,
                            gpu.array.get_pointer(x_gpu), x_gpu.shape, beta,
                            gpu.array.get_pointer(y_gpu), a_gpu.dtype,
                            trans)
        if gpu.debug:
            gpu.debug_test(y_debug, y_gpu, "gemv")
        if y_cpu is not None:
            gpu.copy_to_host(y_gpu, y_cpu)
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if x_cpu is None:
            x_cpu = gpu.copy_to_host(x_gpu)
        if y_cpu is None:
            y_cpu = gpu.copy_to_host(y_gpu)
        _gpaw.gemv(alpha, a_cpu, x_cpu, beta, y_cpu, trans)
        if y_gpu is not None:
            gpu.copy_to_device(y_cpu, y_gpu)


def axpy(alpha, x, y, cuda=None):
    """alpha x plus y.

    Performs the operation::

      y <- alpha * x + y

    """
    if debug:
        if isinstance(alpha, complex):
            assert is_contiguous(x, complex) and is_contiguous(y, complex)
        else:
            assert isinstance(alpha, float)
            assert x.dtype in [float, complex]
            assert x.dtype == y.dtype
            assert x.flags.contiguous and y.flags.contiguous
        assert x.shape == y.shape

    x_cpu, x_gpu = (None, x) if gpu.is_device_array(x) \
                             else (x, None)
    y_cpu, y_gpu = (None, y) if gpu.is_device_array(y) \
                             else (y, None)

    if cuda or (cuda is None and gpu.is_device_array(y)):
        if x_gpu is None:
            x_gpu = gpu.copy_to_device(x_cpu)
        if y_gpu is None:
            y_gpu = gpu.copy_to_device(y_cpu)
        if gpu.debug:
            x_debug = gpu.copy_to_host(x_gpu)
            y_debug = gpu.copy_to_host(y_gpu)
            _gpaw.axpy(alpha, x_debug, y_debug)
        _gpaw.axpy_cuda_gpu(alpha, gpu.array.get_pointer(x_gpu), x_gpu.shape,
                            gpu.array.get_pointer(y_gpu), y_gpu.shape,
                            x_gpu.dtype)
        if gpu.debug:
            gpu.debug_test(y_debug, y_gpu, "axpy")
        if y_cpu is not None:
            gpu.copy_to_host(y_gpu, y_cpu)
    else:
        if x_cpu is None:
            x_cpu = gpu.copy_to_host(x_gpu)
        if y_cpu is None:
            y_cpu = gpu.copy_to_host(y_gpu)
        _gpaw.axpy(alpha, x_cpu, y_cpu)
        if y_gpu is not None:
            gpu.copy_to_device(y_cpu, y_gpu)


def rk(alpha, a, beta, c, trans='c', cuda=None, hybrid=False):
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
    if debug:
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

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    c_cpu, c_gpu = (None, c) if gpu.is_device_array(c) \
                             else (c, None)

    if cuda or (cuda is None and gpu.is_device_array(c)):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if c_gpu is None:
            c_gpu = gpu.copy_to_device(c_cpu)
        if gpu.debug:
            a_debug = gpu.copy_to_host(a_gpu)
            c_debug = gpu.copy_to_host(c_gpu)
            _gpaw.rk(alpha, a_debug, beta, c_debug, trans)
        _gpaw.rk_cuda_gpu(alpha, gpu.array.get_pointer(a_gpu), a_gpu.shape,
                          beta, gpu.array.get_pointer(c_gpu), c_gpu.shape,
                          a_gpu.dtype, hybrid)
        if gpu.debug:
            gpu.debug_test(c_debug, c_gpu, "rk")
        if c_cpu is not None:
            gpu.copy_to_host(c_gpu, c_cpu)
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if c_cpu is None:
            c_cpu = gpu.copy_to_host(c_gpu)
        _gpaw.rk(alpha, a_cpu, beta, c_cpu, trans)
        if c_gpu is not None:
            gpu.copy_to_device(c_cpu, c_gpu)


def r2k(alpha, a, b, beta, c, cuda=None, hybrid=False):
    """Rank-2k update of a matrix.

    Performs the operation::

                        dag        cc       dag
      c <- alpha * a . b    + alpha  * b . a    + beta * c

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
    if debug:
        assert np.isfinite(c).all()
        assert (a.dtype == float and b.dtype == float and c.dtype == float or
                a.dtype == complex and b.dtype == complex and
                c.dtype == complex)
        assert a.flags.contiguous and b.flags.contiguous
        assert a.ndim > 1
        assert a.shape == b.shape
        assert c.shape == (a.shape[0], a.shape[0])
        assert c.strides[1] == c.itemsize

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    b_cpu, b_gpu = (None, b) if gpu.is_device_array(b) \
                             else (b, None)
    c_cpu, c_gpu = (None, c) if gpu.is_device_array(c) \
                             else (c, None)

    if cuda or (cuda is None and gpu.is_device_array(c)):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if b_gpu is None:
            b_gpu = gpu.copy_to_device(b_cpu)
        if c_gpu is None:
            c_gpu = gpu.copy_to_device(c_cpu)
        if gpu.debug:
            a_debug = gpu.copy_to_host(a_gpu)
            b_debug = gpu.copy_to_host(b_gpu)
            c_debug = gpu.copy_to_host(c_gpu)
            _gpaw.r2k(alpha, a_debug, b_debug, beta, c_debug)
        _gpaw.r2k_cuda_gpu(alpha, gpu.array.get_pointer(a_gpu), a_gpu.shape,
                           gpu.array.get_pointer(b_gpu), b_gpu.shape, beta,
                           gpu.array.get_pointer(c_gpu), c_gpu.shape,
                           a_gpu.dtype, hybrid)
        if gpu.debug:
            gpu.debug_test(c_debug, c_gpu, "rk2")
        if c_cpu is not None:
            gpu.copy_to_host(c_gpu, c_cpu)
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if b_cpu is None:
            b_cpu = gpu.copy_to_host(b_gpu)
        if c_cpu is None:
            c_cpu = gpu.copy_to_host(c_gpu)
        _gpaw.r2k(alpha, a_cpu, b_cpu, beta, c_cpu)
        if c_gpu is not None:
            gpu.copy_to_device(c_cpu, c_gpu)


def dotc(a, b):
    """Dot product, conjugating the first vector with complex arguments.

    Returns the value of the operation::

        _
       \   cc
        ) a       * b
       /_  ijk...    ijk...
       ijk...

    ``cc`` denotes complex conjugation.
    """
    if debug:
        assert ((is_contiguous(a, float) and is_contiguous(b, float)) or
                (is_contiguous(a, complex) and is_contiguous(b, complex)))
        assert a.shape == b.shape

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    b_cpu, b_gpu = (None, b) if gpu.is_device_array(b) \
                             else (b, None)

    if gpu.is_device_array(a) or gpu.is_device_array(b):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if b_gpu is None:
            b_gpu = gpu.copy_to_device(b_cpu)
        if gpu.debug:
            a_debug = gpu.copy_to_host(a_gpu)
            b_debug = gpu.copy_to_host(b_gpu)
            cpu = _gpaw.dotc(a_debug, b_debug)
        gpu = _gpaw.dotc_cuda_gpu(gpu.array.get_pointer(a_gpu), a.shape,
                                  gpu.array.get_pointer(b_gpu), a.dtype)
        if gpu.debug:
            gpu.debug_test(cpu, gpu, "dotc")
        return gpu
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if b_cpu is None:
            b_cpu = gpu.copy_to_host(b_gpu)
        return _gpaw.dotc(a_cpu, b_cpu)


def dotu(a, b):
    """Dot product, NOT conjugating the first vector with complex arguments.

    Returns the value of the operation::

        _
       \
        ) a       * b
       /_  ijk...    ijk...
       ijk...


    """
    if debug:
        assert ((is_contiguous(a, float) and is_contiguous(b, float)) or
                (is_contiguous(a, complex) and is_contiguous(b, complex)))
        assert a.shape == b.shape

    a_cpu, a_gpu = (None, a) if gpu.is_device_array(a) \
                             else (a, None)
    b_cpu, b_gpu = (None, b) if gpu.is_device_array(b) \
                             else (b, None)

    if gpu.is_device_array(a) or gpu.is_device_array(b):
        if a_gpu is None:
            a_gpu = gpu.copy_to_device(a_cpu)
        if b_gpu is None:
            b_gpu = gpu.copy_to_device(b_cpu)
        if gpu.debug:
            a_debug = gpu.copy_to_host(a_gpu)
            b_debug = gpu.copy_to_host(b_gpu)
            cpu = _gpaw.dotu(a_debug, b_debug)
        gpu = _gpaw.dotu_cuda_gpu(gpu.array.get_pointer(a_gpu), a.shape,
                                  gpu.array.get_pointer(b_gpu), a.dtype)
        if gpu.debug:
            gpu.debug_test(cpu, gpu, "dotu")
        return gpu
    else:
        if a_cpu is None:
            a_cpu = gpu.copy_to_host(a_gpu)
        if b_cpu is None:
            b_cpu = gpu.copy_to_host(b_gpu)
        return _gpaw.dotu(a_cpu, b_cpu)


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

    def r2k(alpha, a, b, beta, c):  # noqa
        if c.size == 0:
            return
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        c += (alpha * a.reshape((len(a), -1))
              .dot(b.reshape((len(b), -1)).conj().T) +
              alpha * b.reshape((len(b), -1))
              .dot(a.reshape((len(a), -1)).conj().T))

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
