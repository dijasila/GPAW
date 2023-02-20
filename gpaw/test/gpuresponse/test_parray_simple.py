import pytest
from gpaw.gpuresponse.parallelarray import ParallelArrayDescriptor, gemm
from gpaw.mpi import world
import numpy as np
import time
import cupy
#    for distA, distB, distC, NA, NB, NC in [ ((2,2), (4,1), (2,2), 500, 124000, 490),
#                                              ((2,2),  (2,2), (2,2), 826, 166000, 1010) ]:
import cupy
from gpaw.gpu import setup
setup()

from cupy.cuda import memory_hooks
#cupy.cuda.set_allocator(None)
#cupy.cuda.set_pinned_memory_allocator(None)

@pytest.mark.parametrize("M", [10000])
@pytest.mark.parametrize("xp", [cupy])
def test_parray_simple(xp, M):
    if xp is np and M > 9000:
        return
    print(world.size, world.rank,'size and rank', flush=True)
    start = time.time()
    world.barrier()
    end = time.time()
    print(world.rank, 'beginning barrier took', end-start, flush=True)
    full_start = time.time()
    #N1 = 4
    #N2 = 4
    #assert N1*N2 == world.size
    #for distA, distB, distC, NA, NB, NC in [  ((6,6), (6,6), (6,6), 7000, 7000, 7000),
    #                                          ((1,36), (36,1), (6,6), 7000, 7000, 7000),
    #                                          ((36,1), (1,36), (6,6), 7000, 7000, 7000),
    #                                          ((36,1), (1,36), (6,6), 7000, 7000, 7000),
    #                                          ((36,1), (1,36), (36,1), 7000, 7000, 7000)]:
    #for distA, distB, distC, NA, NB, NC in [  ((N1,N2), (N1,N2), (N1,N2), 20000, 20000, 20000)]:
    #for distA, distB, distC, NA, NB, NC in [  ((16,1), (1,16), (4,4), 10000, 10000, 10000)]:
    #for distA, distB, distC, NA, NB, NC in [  ((8,4), (8,4), (8,4), 1500, 1500, 1500)]:
    #for distA, distB, distC, NA, NB, NC in [  ((4,4), (4,4), (4,4), M, M, M)]:
    if 1: # with memory_hooks.DebugPrintHook():
        distA, distB, distC, NA, NB, NC =  ((2,4), (2,4), (2,4), M, M, M)
        shapeA = (NA, NB)
        shapeB = (NB, NC)
        shapeC = (NA, NC)
        def Afun(AX, AY):
            return (AX + 2.*AY)/100
        def Bfun(BX, BY):
            return (3.*BX - 1.1*BY)/100
        def Cfun(CX,CY):
            return 0.*CX

        if 1: # world.rank == 0:
            # Reference
            AX, AY = xp.indices(shapeA)
            A = Afun(AX,AY)
            del AX
            del AY
            BX, BY = xp.indices(shapeB)
            B = Bfun(BX, BY)
            del BX
            del BY

            if xp is cupy:
                cupy.cuda.runtime.deviceSynchronize()
            #s1 = xp.cuda.Stream(non_blocking=True)
            #s2 = xp.cuda.Stream(non_blocking=True)
            #s3 = xp.cuda.Stream(non_blocking=True)
            #s4 = xp.cuda.Stream(non_blocking=True)
            print('Starting serial gemm', flush=True)
            C = A @ B  # For good timing, do it first once
            start = time.time()
            C = A @ B 
 
            if xp is cupy:
                cupy.cuda.runtime.deviceSynchronize()
            xp.asnumpy(C)
            end = time.time()
            print('Serial gemm took', end-start, flush=True)
            serial_Tflops = NA*(2*NB-1)*NC / (end-start) / 1000**4
            print(world.rank, 'Serial performance', serial_Tflops, 'Tflops', M, xp.__name__, flush=True)
        start = time.time()
        padA = ParallelArrayDescriptor(shapeA, comm=world, xp=xp)
        padB = ParallelArrayDescriptor(shapeB, comm=world, xp=xp)
        padC = ParallelArrayDescriptor(shapeC, comm=world, xp=xp)
        pA = padA.create_from_function(Afun, distA)
        pB = padB.create_from_function(Bfun, distB)
        pC = padC.create_from_function(Cfun, distC)
        #pA.fill(world.rank+1)
        #pB.fill((world.rank+1)*10)
        #pC.fill(0)
        if xp is cupy:
            cupy.cuda.runtime.deviceSynchronize()
        world.barrier()
        end = time.time()
        print('Building arrays took', end-start, flush=True)
        start = time.time()
        for i in range(1):
            single_start = time.time() 
            gemm(pA, pB, pC)
            single_end = time.time()
            print(world.rank, 'single gemm took iter', i, single_end-single_start)
            if xp is cupy:
                mempool = xp.get_default_memory_pool()
                mempool.free_all_blocks()
                cupy.cuda.runtime.deviceSynchronize()
        world.barrier()
        end = time.time()
        parallel_Tflops = 1*NA*(2*NB+1)*NC / (end-start) / 1000**4
        print(world.rank, 'Parallel gemm took', end-start, flush=True)
        print(world.rank, 'Parallel performance', parallel_Tflops, 'Tflops', M, xp.__name__, flush=True)
        full_end = time.time()
        print(world.rank, 'full function took', full_end-full_start, flush=True)
        if 1: #world.rank == 0:
            print(world.rank, 'Speedup', distA, distB, distC, parallel_Tflops/serial_Tflops, xp.__name__, flush=True)
        #A = pA.collect()
        #B = pB.collect()
        #C = xp.asnumpy(A @ B)
         
        if xp is cupy:
            Cnew = xp.asnumpy(pC.collect())
        else:
            Cnew = pC.collect()
        if world.rank == 0:
            if xp is cupy:
                C = xp.asnumpy(C)
            fails = 0
            start = time.time()
            end = time.time()
            print(world.rank, 'Collect took', end-start, flush=True)
            start = time.time()
            diff = np.max(np.abs(Cnew-C).ravel())
            print('Maximum absolute deviation in matrix element', diff)
            if not np.allclose(Cnew, C, rtol=1e-5, atol=1e-3):
                print('Assert failed, iterating failures', flush=True)
                if world.rank == 0:
                    np.save('Cnew.npy', Cnew)
                    np.save('C.npy', C)
                    for i, (C1, C2) in enumerate(zip(Cnew.ravel(), C.ravel())):
                        if np.abs(C1-C2)>1e-3:
                            print(i, C1, C2, flush=True)
                            fails += 1
                        if fails > 10000:
                            break
            else:
                print('Assert passed')
            #assert np.allclose(Cnew, C, rtol=1e-3, atol=1)
            #if world.rank == 0:
            #    with open(f'gemm_{NA}_{NB}_{NC}.txt','w') as f:
            #        for C1, C2 in zip(Cnew.ravel(), C.ravel()):
            #            print(C1, C2, file=f)
            #world.barrier()
            end = time.time()
            print(world.rank, 'Allclose took', end-start, flush=True)
        

