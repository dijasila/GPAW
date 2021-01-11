"""Test of PBLAS Level 2 & 3 : rk, r2k, gemv, gemm.

The test generates random matrices A0, B0, X0, etc. on a
1-by-1 BLACS grid. They are redistributed to a mprocs-by-nprocs
BLACS grid, BLAS operations are performed in parallel, and
results are compared against BLAS.
"""

import pytest
import numpy as np

from gpaw.mpi import world, rank, broadcast_float
from gpaw.test import equal
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.utilities import compiled_with_sl
from gpaw.utilities.blas import gemm, r2k, rk
from gpaw.utilities.scalapack import pblas_simple_gemm, pblas_simple_gemv, \
    pblas_simple_r2k, pblas_simple_rk, \
    pblas_simple_hemm, pblas_simple_symm, pblas_hemm, pblas_symm

pytestmark = pytest.mark.skipif(not compiled_with_sl(),
                                reason='not compiled with scalapack')

# may need to be be increased if the mprocs-by-nprocs
# BLACS grid becomes larger
tol = 4.0e-13

mnprocs_i = [(1, 1)]
if world.size >= 2:
    mnprocs_i += [(1, 2), (2, 1)]
if world.size >= 4:
    mnprocs_i += [(2, 2)]
if world.size >= 8:
    mnprocs_i += [(2, 4), (4, 2)]


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('dtype', [float, complex])
def test_parallel_pblas(dtype, mprocs, nprocs,
                        M=160, N=120, K=140, seed=42):
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, mprocs, nprocs)

    if dtype == complex:
        epsilon = 1.0j
    else:
        epsilon = 0.0

    # Create descriptors for matrices on master:
    globA = grid.new_descriptor(M, K, M, K)
    globB = grid.new_descriptor(K, N, K, N)
    globC = grid.new_descriptor(M, N, M, N)
    globZ = grid.new_descriptor(K, K, K, K)
    globX = grid.new_descriptor(K, 1, K, 1)
    globY = grid.new_descriptor(M, 1, M, 1)
    globD = grid.new_descriptor(M, K, M, K)
    globS = grid.new_descriptor(M, M, M, M)
    globU = grid.new_descriptor(M, M, M, M)

    # print globA.asarray()
    # Populate matrices local to master:
    A0 = gen.rand(*globA.shape) + epsilon * gen.rand(*globA.shape)
    B0 = gen.rand(*globB.shape) + epsilon * gen.rand(*globB.shape)
    D0 = gen.rand(*globD.shape) + epsilon * gen.rand(*globD.shape)
    X0 = gen.rand(*globX.shape) + epsilon * gen.rand(*globX.shape)

    # Local result matrices
    Y0 = globY.empty(dtype=dtype)
    C0 = globC.zeros(dtype=dtype)
    Z0 = globZ.zeros(dtype=dtype)
    S0 = globS.zeros(dtype=dtype)  # zeros needed for rank-updates
    U0 = globU.zeros(dtype=dtype)  # zeros needed for rank-updates

    # Local reference matrix product:
    if rank == 0:
        # C0[:] = np.dot(A0, B0)
        gemm(1.0, B0, A0, 0.0, C0)
        # gemm(1.0, A0, A0, 0.0, Z0, transa='t')
        print(A0.shape, Z0.shape)
        Z0[:] = np.dot(A0.T, A0)
        # Y0[:] = np.dot(A0, X0)
        # gemv(1.0, A0, X0.ravel(), 0.0, Y0.ravel())
        Y0[:, 0] = A0.dot(X0.ravel())
        r2k(1.0, A0, D0, 0.0, S0)
        rk(1.0, A0, 0.0, U0)
    assert globA.check(A0) and globB.check(B0) and globC.check(C0)
    assert globX.check(X0) and globY.check(Y0)
    assert globD.check(D0) and globS.check(S0) and globU.check(U0)

    # Create distributed destriptors with various block sizes:
    distA = grid.new_descriptor(M, K, 2, 2)
    distB = grid.new_descriptor(K, N, 2, 4)
    distC = grid.new_descriptor(M, N, 3, 2)
    distZ = grid.new_descriptor(K, K, 5, 7)
    distX = grid.new_descriptor(K, 1, 4, 1)
    distY = grid.new_descriptor(M, 1, 3, 1)
    distD = grid.new_descriptor(M, K, 2, 3)
    distS = grid.new_descriptor(M, M, 2, 2)
    distU = grid.new_descriptor(M, M, 2, 2)

    # Distributed matrices:
    A = distA.empty(dtype=dtype)
    B = distB.empty(dtype=dtype)
    C = distC.empty(dtype=dtype)
    Z = distZ.empty(dtype=dtype)
    X = distX.empty(dtype=dtype)
    Y = distY.empty(dtype=dtype)
    D = distD.empty(dtype=dtype)
    S = distS.zeros(dtype=dtype)  # zeros needed for rank-updates
    U = distU.zeros(dtype=dtype)  # zeros needed for rank-updates
    Redistributor(world, globA, distA).redistribute(A0, A)
    Redistributor(world, globB, distB).redistribute(B0, B)
    Redistributor(world, globX, distX).redistribute(X0, X)
    Redistributor(world, globD, distD).redistribute(D0, D)

    pblas_simple_gemm(distA, distB, distC, A, B, C)
    pblas_simple_gemm(distA, distA, distZ, A, A, Z, transa='T')
    pblas_simple_gemv(distA, distX, distY, A, X, Y)
    pblas_simple_r2k(distA, distD, distS, A, D, S)
    pblas_simple_rk(distA, distU, A, U)

    # Collect result back on master
    C1 = globC.empty(dtype=dtype)
    Y1 = globY.empty(dtype=dtype)
    S1 = globS.zeros(dtype=dtype)  # zeros needed for rank-updates
    U1 = globU.zeros(dtype=dtype)  # zeros needed for rank-updates
    Redistributor(world, distC, globC).redistribute(C, C1)
    Redistributor(world, distY, globY).redistribute(Y, Y1)
    Redistributor(world, distS, globS).redistribute(S, S1)
    Redistributor(world, distU, globU).redistribute(U, U1)

    if rank == 0:
        gemm_err = abs(C1 - C0).max()
        gemv_err = abs(Y1 - Y0).max()
        r2k_err = abs(S1 - S0).max()
        rk_err = abs(U1 - U0).max()
        print('gemm err', gemm_err)
        print('gemv err', gemv_err)
        print('r2k err', r2k_err)
        print('rk_err', rk_err)
    else:
        gemm_err = 0.0
        gemv_err = 0.0
        r2k_err = 0.0
        rk_err = 0.0

    gemm_err = world.sum(gemm_err)  # We don't like exceptions on only one cpu
    gemv_err = world.sum(gemv_err)
    r2k_err = world.sum(r2k_err)
    rk_err = world.sum(rk_err)

    equal(gemm_err, 0, tol)
    equal(gemv_err, 0, tol)
    equal(r2k_err, 0, tol)
    equal(rk_err, 0, tol)


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('uplo', ['L', 'U'])
@pytest.mark.parametrize('side', ['L', 'R'])
@pytest.mark.parametrize('simple', [True, False])
@pytest.mark.parametrize('hemm', [True, False])
@pytest.mark.parametrize('dtype', [float, complex])
def test_pblas_hemm_symm(dtype, hemm, simple, uplo, side, mprocs, nprocs,
                         M=160, N=120, seed=42):
    """Test pblas_simple_hemm, pblas_simple_symm, pblas_hemm, pblas_symm

    The operation is
    * C <- alpha*A*B + beta*C  if side == 'L'
    * C <- alpha*B*A + beta*C  if side == 'R'

    The computations are done with
    * lower triangular of A    if uplo == 'L'
    * upper triangular of A    if uplo == 'U'

    Additional options
    * A is Hermitian           if hemm == True
    * A is symmetric           if hemm == False
    * alpha=1 and beta=0       if simple == True
    """
    gen = np.random.default_rng(seed)
    grid = BlacsGrid(world, mprocs, nprocs)

    if dtype == complex:
        def random(*args):
            return gen.random(*args) + 1.0j * gen.random(*args)
    else:
        def random(*args):
            return gen.random(*args)

    # Create descriptors for matrices
    if side == 'L':
        globA = grid.new_descriptor(M, M, M, M)
        distA = grid.new_descriptor(M, M, 2, 2)
    else:
        globA = grid.new_descriptor(N, N, N, N)
        distA = grid.new_descriptor(N, N, 2, 2)
    globB = grid.new_descriptor(M, N, M, N)
    distB = grid.new_descriptor(M, N, 2, 4)
    globC = grid.new_descriptor(M, N, M, N)
    distC = grid.new_descriptor(M, N, 3, 2)

    # Generate random matrices and coefficients
    if simple:
        alpha = 1.0
        beta = 0.0
    else:
        alpha = random()
        beta = random()
    A0 = random(globA.shape)
    B0 = random(globB.shape)
    C0 = random(globC.shape)

    if world.rank == 0:
        # Prepare A matrix
        if hemm:
            # Hermitian matrix
            A0 = A0 + A0.T.conj()
        else:
            # Symmetric matrix
            A0 = A0 + A0.T
        A0 = np.ascontiguousarray(A0)

        # Local reference matrix product
        if side == 'L':
            ref_C0 = alpha * np.dot(A0, B0) + beta * C0
        else:
            ref_C0 = alpha * np.dot(B0, A0) + beta * C0

        # Only lower or upper triangular is used, so
        # fill the other triangular with NaN to detect errors
        if uplo == 'L':
            A0 += np.triu(A0 * np.nan, 1)
        else:
            A0 += np.tril(A0 * np.nan, -1)

        if world.rank == 0:
            print(A0)

    assert globA.check(A0)
    assert globB.check(B0)
    assert globC.check(C0)

    # Distribute matrices
    A = distA.empty(dtype=dtype)
    B = distB.empty(dtype=dtype)
    C = distC.empty(dtype=dtype)
    Redistributor(world, globA, distA).redistribute(A0, A)
    Redistributor(world, globB, distB).redistribute(B0, B)
    Redistributor(world, globC, distC).redistribute(C0, C)

    # Calculate with scalapack
    if simple and hemm:
        pblas_simple_hemm(distA, distB, distC, A, B, C,
                          uplo=uplo, side=side)
    elif hemm:
        pblas_hemm(alpha, A, B, beta, C, distA, distB, distC,
                   uplo=uplo, side=side)
    elif simple:
        pblas_simple_symm(distA, distB, distC, A, B, C,
                          uplo=uplo, side=side)
    else:
        pblas_symm(alpha, A, B, beta, C, distA, distB, distC,
                   uplo=uplo, side=side)

    # Collect result back on master
    Redistributor(world, distC, globC).redistribute(C, C0)
    if world.rank == 0:
        err = np.abs(ref_C0 - C0).max()
    else:
        err = np.nan

    err = broadcast_float(err, world)
    assert err < tol
