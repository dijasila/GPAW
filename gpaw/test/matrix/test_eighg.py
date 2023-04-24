import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import broadcast_exception, world
import scipy.linalg as linalg


@pytest.mark.parametrize('dtype', [float, complex])
def test_eighg(dtype):
    n = 5
    S0 = Matrix(n, n, dist=(world, 1, 1), dtype=dtype)
    S = S0.new(dist=(world, world.size, 1))
    H0 = S0.new()
    H = S.new()
    if world.rank == 0:
        S0.data[:] = np.eye(n)
        H0.data[:] = 0.0
        H0.data.ravel()[::n + 1] = np.arange(n) + 1
        if dtype == float:
            S0.data[-1, 0] = 0.001
            H0.data[-1, 0] = 0.001
        else:
            S0.data[-1, 0] = 0.001j
            H0.data[-1, 0] = 0.001j
    H0.tril2full()
    S0.tril2full()
    H00 = H0.copy()
    S0.redist(S)
    H0.redist(H)
    L = S.copy()
    L.invcholesky()
    if world.rank == 0:
        eigs0, C0 = linalg.eigh(H0.data, S0.data)
        print(eigs0)
        print(C0)
        error = H0.data @ C0 - S0.data @ C0 @ np.diag(eigs0)
        print(error)

    L0 = S0.new()
    L.redist(L0)
    eigs = H.eighg(L0)
    H.redist(H0)
    print(eigs)
    print(H0.data)
    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(eigs - eigs0).max() < 1e-14
            error = H00.data @ H0.data - S0.data @ H0.data @ np.diag(eigs)
            assert abs(error).max() < 1e-14
