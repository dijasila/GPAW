import numpy as np
import pytest
from ase.parallel import world
from gpaw.utilities import compiled_with_sl
from gpaw.eigensolvers.diagonalizerbackend import (
    ScipyDiagonalizer,
    ScalapackDiagonalizer,
)


def prepare_eigensolver_matrices(size_of_matrices, dtype):

    matrix_dimensions = [size_of_matrices, size_of_matrices]
    rng = np.random.Generator(np.random.PCG64(24589246))
    A = rng.random(matrix_dimensions).astype(dtype)
    B = rng.random(matrix_dimensions).astype(dtype)

    if dtype == complex:
        A += 1j * rng.random(matrix_dimensions)
        B += 1j * rng.random(matrix_dimensions)
    A = A + A.T.conj()
    B = B + B.T.conj()
    # Make sure B is positive definite
    B += np.eye(size_of_matrices) * size_of_matrices

    return A, B


@pytest.mark.parametrize('dtype,', [float, complex])
def test_scipy_diagonalizer_eigenproblem_correctness(dtype):
    is_master_rank = world.rank == 0
    eigenproblem_size = 64

    serial_diagonalizer = ScipyDiagonalizer()

    a, b = prepare_eigensolver_matrices(eigenproblem_size, dtype=dtype)
    eps = np.zeros(eigenproblem_size)
    a_scipy_copy = a.copy()
    b_scipy_copy = b.copy()

    # a_scipy_copy contains eigenvectors after this.
    serial_diagonalizer.diagonalize(
        a_scipy_copy,
        b_scipy_copy,
        eps,
        is_gridband_master=is_master_rank,
        debug=False,
    )

    if is_master_rank:
        assert np.allclose(a @ a_scipy_copy, b @ a_scipy_copy @ np.diag(eps))


@pytest.mark.parametrize('dtype,', [float, complex])
@pytest.mark.skipif(
    not compiled_with_sl(),
    reason='Not compiled with Scalapack',
)
def test_diagonalizer_eigenproblem_correctness(dtype):

    is_master_rank = world.rank == 0
    eigenproblem_size = world.size * 64

    nrows = 2 if world.size > 1 else 1
    ncols = world.size // 2 if nrows > 1 else 1

    scalapack_diagonalizer = ScalapackDiagonalizer(
        eigenproblem_size,
        nrows,
        ncols,
        dtype=dtype,
        scalapack_communicator=world,
        blocksize=32 if world.size == 1 else 64,
    )

    a, b = prepare_eigensolver_matrices(eigenproblem_size, dtype=dtype)
    eps = np.zeros(eigenproblem_size)

    a_scalapack_copy = a.copy()
    b_scalapack_copy = b.copy()

    # a_scalapack_copy contains eigenvectors after this.
    scalapack_diagonalizer.diagonalize(
        a_scalapack_copy,
        b_scalapack_copy,
        eps,
        is_gridband_master=is_master_rank,
        debug=False,
    )

    if is_master_rank:
        assert np.allclose(
            a @ a_scalapack_copy, b @ a_scalapack_copy @ np.diag(eps)
        )
