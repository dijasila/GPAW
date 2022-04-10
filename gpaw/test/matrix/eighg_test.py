import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import broadcast_exception, world
import scipy.linalg as linalg


#@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('dtype', [complex])
@pytest.mark.skipif(world.size > 2, reason='size>2')
def test_invcholesky(dtype):
    n = 2
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
            H0.data[0, -1] = 0.001
        else:
            S0.data[-1, 0] = 0.001j
            H0.data[-1, 0] = 0.001j
            H0.data[0, -1] = -0.001j
    S0.redist(S)
    H0.redist(H)
    L = S.copy()
    L.invcholesky()
    eigs = H.eighg(L)

    S.tril2full()
    if world.size == 1:
        print(L.data @ S.data @ L.data.T.conj())
        e,c=linalg.eigh(H.data, S.data)
        print(0)
        print(H.data @ c - S.data @ c @ np.diag(e))
        print(e,c)
            print(H.data)
    else:
        print(world.rank, L.data)
    A = L.multiply(S, opa='N').multiply(L, opb='C')
    #print(A.data)
    A.redist(S0)
    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(S0.data - np.eye(2)).max() < 1e-14
