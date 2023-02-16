import time
start = time.time()
import cupy as cp
import numpy as np
from gpaw.mpi import world
from gpaw.gpu import setup
setup()
stop = time.time()
print('Imports took', stop-start, world.rank)
#cp.cuda.set_allocator(cp.cuda.malloc_managed)
for n in range(3):
    for xp in [cp, np]:
        for dtype in [float]:
            A = xp.ones((5000, 5000), dtype=dtype) * (world.rank + 1)
            start = time.time()
            for i in range(5):
                world.sum(A, root=0)
            stop = time.time()
            if world.rank == 0:
                print(xp, 'sum (handwaving speed)', np.log2(world.size)*5000**2*8*8*5/1024**3/(stop-start), 'GB/s', flush=True)
                

            if world.rank < 8:
                A = xp.ones((15000,15000), dtype=dtype) * (world.rank + 1)
                cp.cuda.runtime.deviceSynchronize()
                world.barrier()
                for i in range(30):
                    world.send(A, world.rank+8)
                world.barrier()
            else:
                total_data = 15000**2*30*8
                A = xp.empty((15000,15000), dtype=dtype)
                cp.cuda.runtime.deviceSynchronize()
                world.barrier()
                start = time.time()
                for i in range(30):
                    world.receive(A, world.rank-8)
                end = time.time()
                world.barrier()
                if world.rank == 8:
                    print(xp, total_data*8/(end-start) / 1024**3, 'GB/s', flush=True)
                assert xp.all(A == (world.rank + 1 - 8))
            """
            start = time.time()
            B = A+A
            cp.cuda.runtime.deviceSynchronize()
            stop = time.time()
            print('Simple memory read took', stop-start, flush=True)
            if world.rank == 0:
                print(xp, A.dtype.itemsize, A.dtype.num,  ' dtypenum', flush=True)
            if xp == cp:  # Let's not get ahead of ourselves
                cp.cuda.runtime.deviceSynchronize()
            start = time.time()
            world.sum(A)
            if xp == cp:  # Let's not get ahead of ourselves
                cp.cuda.runtime.deviceSynchronize()
            stop = time.time()            
            if world.rank == 0:
                #if xp == cp:
                #    A = cp.asnumpy(A)
                #print(A)
                assert xp.all(A == (world.size+1)*world.size//2)
                print('Success', stop-start, 's', flush=True)
            """
