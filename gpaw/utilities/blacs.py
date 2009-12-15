# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the C and Fortran packages:
Basic Linear Algebra Communication Subprogramcs (BLACS)
ScaLAPACK

See also:
http://www.netlib.org/blacs
and
http://www.netlib.org/scalapack
"""

from sys import stderr

import numpy as np

from gpaw.utilities import warning
from gpaw import debug
import gpaw.mpi as mpi
import _gpaw

def scalapack_diagonalize_dc(desca, a, z, w, uplo):
    if not desca.blacsgrid.is_active():
        return    
    assert desca.check(a)
    assert desca.check(w)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    # stil need assert for eigenvalues
    assert uplo in ['L', 'U']
    _gpaw.scalapack_diagonalize_dc(a, desca.asarray(), uplo, z, w)

def scalapack_diagonalize_ex(desca, a, z, w, uplo):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a)
    assert desca.check(w)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    # stil need assert for eigenvalues
    assert uplo in ['L', 'U']
    _gpaw.scalapack_diagonalize_ex(a, desca.asarray(), uplo, z, w)

def scalapack_general_diagonalize_ex(desca, a, b, z, w, uplo):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a)
    assert desca.check(b)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    _gpaw.scalapack_general_diagonalize_ex(a, desca.asarray(), 
                                           uplo, b, z, w)


def scalapack_inverse_cholesky(desca, a, uplo):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    _gpaw.scalapack_inverse_cholesky(a, desca.asarray(), uplo)


def pblas_gemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
                 transa='N', transb='N'):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a_MK)
    assert descb.check(b_KN)
    assert descc.check(c_MN)
    assert transa in ['N', 'T', 'C'] and transb in ['N', 'T', 'C']
    M, K = desca.gshape
    K, N = descb.gshape
    if transb == 'N':
        assert desca.gshape[1] == descb.gshape[0]
        assert desca.gshape[0] == descc.gshape[0]
        assert descb.gshape[1] == descc.gshape[1]
    if transb == 'T':
        N, K = K, N
        assert desca.gshape[1] == descb.gshape[1]
        assert desca.gshape[0] == descc.gshape[0]
        assert descb.gshape[0] == descc.gshape[1]
    assert transa == 'N' # XXX remember to implement 'T'
    _gpaw.pblas_gemm(N, M, K, alpha, b_KN.T, a_MK.T, beta, c_MN.T,
                     descb.asarray(), desca.asarray(), descc.asarray(),
                     transb, transa)


def pblas_simple_gemm(desca, descb, descc, a_MK, b_KN, c_MN, 
                      transa='N', transb='N'):
    alpha = 1.0
    beta = 0.0
    pblas_gemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
               transa, transb)


def pblas_gemv(alpha, a, x, beta, y, desca, descx, descy,
               transa='T'):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a)
    assert descx.check(x)
    assert descy.check(y)
    M, N = desca.gshape
    # XXX transa = 'N' not implemented
    assert transa in ['T', 'C']
    assert desca.gshape[0] == descy.gshape[0]
    assert desca.gshape[1] == descx.gshape[0]
    assert descx.gshape[1] == descy.gshape[1]
    _gpaw.pblas_gemv(N, M, alpha,
                     a, x, beta, y,
                     desca.asarray(),
                     descx.asarray(),
                     descy.asarray(), 
                     transa)


def pblas_simple_gemv(desca, descx, descy, a, x, y):
    alpha = 1.0
    beta = 0.0
    pblas_gemv(alpha, a, x, beta, y, desca, descx, descy)

def pblas_r2k(alpha, a_NK, b_NK, beta, c_NN, desca, descb, descc,
                uplo='U'):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a_NK)
    assert descb.check(b_NK)
    assert descc.check(c_NN)
    assert descc.gshape[0] == descc.gshape[1] # symmetric matrix
    assert desca.gshape == descb.gshape # same shape
    assert uplo in ['L', 'U']
    N = descc.gshape[0] # order of C
    # K must take into account implicit tranpose due to C ordering
    K = desca.gshape[1] # number of columns of A and B
    _gpaw.pblas_r2k(N, K, alpha, a_NK, b_NK, beta, c_NN,
                    desca.asarray(), 
                    descb.asarray(), 
                    descc.asarray(),
                    uplo)


def pblas_simple_r2k(desca, descb, descc, a, b, c):
    alpha = 1.0
    beta = 0.0
    pblas_r2k(alpha, a, b, beta, c, 
                desca, descb, descc)

def pblas_rk(alpha, a_NK, beta, c_NN, desca, descc,
             uplo='U'):
    if not desca.blacsgrid.is_active():
        return
    assert desca.check(a_NK)
    assert descc.check(c_NN)
    assert descc.gshape[0] == descc.gshape[1] # symmetrix matrix
    assert uplo in ['L', 'U']
    N = descc.gshape[0] # order of C
    # K must take into account implicit tranpose due to C ordering
    K = desca.gshape[1] # number of columns of A
    _gpaw.pblas_rk(N, K, alpha, a_NK, beta, c_NN,
                    desca.asarray(), 
                    descc.asarray(),
                    uplo)


def pblas_simple_rk(desca, descc, a, c):
    alpha = 1.0 
    beta = 0.0 
    pblas_rk(alpha, a, beta, c, 
             desca, descc)

#if not debug:
#    blacs_create = _gpaw.blacs_create
#    blacs_destroy = _gpaw.blacs_destroy
#    scalapack_redist1 = _gpaw.scalapack_redist1
#    scalapack_diagonalize_dc = _gpaw.scalapack_diagonalize_dc
#    scalapack_diagonalize_ex = _gpaw.scalapack_diagonalize_ex
#    scalapack_inverse_cholesky = _gpaw.scalapack_inverse_cholesky
