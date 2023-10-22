import pytest
import numpy as np
from _gpaw import multi_einsum_gpu
from time import time

def multi_einsum_np(path, args, out):
    for i in range(len(args)):
        np.einsum(path, *args[i], out=out[i])
        print(out[i].shape)

def multi_einadd_np(path, args, out):
    for i in range(len(args)):
        out[i] += np.einsum(path, *args[i])
        print(out[i].shape)
import cupy as cp

@pytest.mark.gpu
@pytest.mark.parametrize('add', [True, False])
@pytest.mark.parametrize('na', [0, 1, 100])
def test_multi_einsum(add, na):
    nn = 300
    ni_a = np.random.randint(30, 50, size=na)
    P_ani = [ np.random.rand(nn, ni) for ni in ni_a]
    f_n = np.random.rand(nn)
    D_aii = [ np.random.rand(ni, ni) for ni in ni_a ]
    D_aii_old = [ D_ii.copy() for D_ii in D_aii ]
    f_an = [ f_n ] * na
    start = time()
    if add:
        multi_einadd_np('ni,n,nj->ij', list(zip(P_ani, f_an, P_ani)), out=D_aii)
    else:
        multi_einsum_np('ni,n,nj->ij', list(zip(P_ani, f_an, P_ani)), out=D_aii)
    stop = time()
    print('numpy einsum took', stop-start)

    P_ani_gpu = [ cp.asarray(P_ni) for P_ni in P_ani ]
    f_an_gpu = [ cp.asarray(f_n) ] * na

    if add:
        D_aii_gpu = [ cp.asarray(D_ii) for D_ii in D_aii_old ]
    else:
        D_aii_gpu = [ cp.zeros_like(D_ii) for D_ii in D_aii_old ]
    cp.cuda.runtime.deviceSynchronize()
    start = time()
    
    if add:
        multi_einsum_gpu('ni,n,nj->ij', list(zip(P_ani_gpu, f_an_gpu, P_ani_gpu)), add=D_aii_gpu)
    else:
        multi_einsum_gpu('ni,n,nj->ij', list(zip(P_ani_gpu, f_an_gpu, P_ani_gpu)), out=D_aii_gpu)
    cp.cuda.runtime.deviceSynchronize()
    stop = time()
    print('GPU einsum took', stop-start)

    for D_ii, D_ii_gpu in zip(D_aii, D_aii_gpu):
        assert cp.allclose(D_ii, D_ii_gpu)
