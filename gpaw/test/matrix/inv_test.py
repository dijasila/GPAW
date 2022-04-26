import numpy as np
import pytest
from gpaw.core.matrix import Matrix, _global_blacs_context_store
from gpaw.mpi import broadcast_exception, world


@pytest.mark.parametrize('dtype', [float, complex])
# @pytest.mark.skipif(world.size > 2, reason='size>2')
def test_inv(dtype):
    N = 5
    S0 = Matrix(N, N,
                dist=(world, 1, 1),
                dtype=dtype)
    S = S0.new(dist=(world, world.size, 1))

    print('S0', S0.dist.desc)
    if world.rank == 0:
        S0.data[:] = np.diag(np.arange(1, N + 1))
        if dtype == float:
            S0.data[-1, 0] = 0.1
        else:
            S0.data[-1, 0] = 0.1j

    print('S', S.dist.desc, _global_blacs_context_store)
    S0.redist(S)

    iS = S.copy()
    iS.inv()
    return
    S.tril2full()
    iS.tril2full()
    print('s', S.data @ iS.data)
    A = S.multiply(iS)
    print(A.data)
    A.redist(S0)
    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(S0.data - np.eye(N)).max() < 1e-14
