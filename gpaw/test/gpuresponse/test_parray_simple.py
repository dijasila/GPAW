from gpaw.gpuresponse.parallelarray import ParallelArrayDescriptor, gemm
from gpaw.mpi import world
import numpy as np
import time

def test_parray_simple():
    print(world.size, world.rank,'size and rank')

    dist = (4,10)
    A, B, C = 2560, 2000, 2999
    shapeA = (A, B)
    shapeB = (B, C)
    shapeC = (A, C)
    # Reference
    AX, AY = np.indices(shapeA)
    BX, BY = np.indices(shapeB)
    A = AX + 2.*AY
    B = 3.*BX - 1.1*BY
    start = time.time()   
    C = A @ B
    end = time.time()
    print('Serial gemm took', end-start)

    padA = ParallelArrayDescriptor(shapeA, comm=world, xp=np)
    padB = ParallelArrayDescriptor(shapeB, comm=world, xp=np)
    padC = ParallelArrayDescriptor(shapeC, comm=world, xp=np)
    start = time.time()
    pA = padA.create_from_global(A, dist)
    pB = padB.create_from_global(B, dist)
    pC = padC.create_from_global(C, dist)
    pC.fill(0)
    end = time.time()
    print('Building arrays took', end-start)
    start = time.time()   
    gemm(pA, pB, pC)
    end = time.time()
    print('Parallel gemm took', end-start)
    start = time.time()
    Cnew = pC.collect()
    end = time.time()
    print('Collect took', end-start)
    start = time.time()
    assert np.allclose(Cnew, C)
    end = time.time()
    print('Allclose took', end-start)

