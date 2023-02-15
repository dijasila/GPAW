import cupy as cp
import numpy as np

from gpaw.mpi import world
print('Running with size', world.size)
from gpaw.gpu import setup

setup()
for xp in [np, cp]:
    for dtype in [float, complex]:
        A = xp.ones((123,132), dtype=dtype) * (world.rank + 1)
        if world.rank == 0:
            print(xp, A.dtype.itemsize, A.dtype.num, type(A.dtype.itemsize), ' dtypenum', flush=True)
        if xp == cp:  # Let's not get ahead of ourselves
            cp.cuda.runtime.deviceSynchronize()
        world.sum(A)
        if world.rank == 0:
            print(A)
            if xp == cp:
                A = cp.asnumpy(A)
            print(A)
            assert np.all(A == (world.size+1)*world.size//2)
            print('Success', flush=True)

