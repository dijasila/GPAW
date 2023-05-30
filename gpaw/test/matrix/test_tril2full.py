import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import broadcast_exception, world


@pytest.mark.parametrize('dtype', [float, complex])
def test_tril2full(dtype):
    N = 21
    S0 = Matrix(N, N, dist=(world, 1, 1), dtype=dtype)
    S = S0.new(dist=(world, world.size, 1))
    if world.rank == 0:
        S0.data[:] = np.random.random((N, N))
        if dtype == complex:
            S0.data.imag[:] = np.random.random((N, N))
            S0.data.ravel().imag[::N + 1] = 0.0
    S0.redist(S)
    B = S.copy()
    S.tril2full()
    S.redist(S0)
    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(S0.data - S0.data.T.conj()).max() == 0.0
    B.add_hermitian_conjugate(0.5)
    B.redist(S0)
    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(S0.data - S0.data.T.conj()).max() == 0.0
