import pytest
import numpy as np
from _gpaw import multi_einsum_gpu
from time import time

def multi_einsum_np(path, args, out):
    for i in range(len(args)):
        np.einsum(path, *args[i], out=out[i], optimize=True)
        print(out[i].shape)

def multi_einadd_np(path, args, out):
    for i in range(len(args)):
        out[i] += np.einsum(path, *args[i], optimize=True)
        print(out[i].shape)
import cupy as cp

def rand(*args, dtype=None):
    if dtype == float:
        return np.random.rand(*args)
    return np.random.rand(*args) +1j * np.random.rand(*args)

@pytest.mark.gpu
@pytest.mark.parametrize('add', [True, False])
@pytest.mark.parametrize('na', [0,1,100])
@pytest.mark.parametrize('dtype', [float, np.complex128])
@pytest.mark.parametrize('cc', [True, False])
def test_multi_einsum(add, na, dtype, cc):
    nn = 300
    ni_a = np.random.randint(30, 50, size=na)
    P_ani = [ rand(nn, ni, dtype=dtype) for ni in ni_a]
    f_n = rand(nn, dtype=dtype)
    D_aii = [ rand(ni, ni, dtype=dtype) for ni in ni_a ]
    D_aii_old = [ D_ii.copy() for D_ii in D_aii ]
    f_an = [ f_n ] * na
    einsum_str = "ni,n,nj->ij"
    einsum_gpustr = f"ni{'*' if cc else ''},n,nj->ij"
    def maybecc(a):
        if cc:
            a = [ aa.conj() for aa in a]
        return a
    start = time()
    if add:
        multi_einadd_np(einsum_str, list(zip(maybecc(P_ani), f_an, P_ani)), out=D_aii)
    else:
        multi_einsum_np(einsum_str, list(zip(maybecc(P_ani), f_an, P_ani)), out=D_aii)
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
        multi_einsum_gpu(einsum_gpustr, list(zip(P_ani_gpu, f_an_gpu, P_ani_gpu)), add=D_aii_gpu)
    else:
        multi_einsum_gpu(einsum_gpustr, list(zip(P_ani_gpu, f_an_gpu, P_ani_gpu)), out=D_aii_gpu)
    cp.cuda.runtime.deviceSynchronize()
    stop = time()
    print('GPU einsum took', stop-start)

    for D_ii, D_ii_gpu in zip(D_aii, D_aii_gpu):
        print(D_ii)
        print(D_ii_gpu)
        assert cp.allclose(D_ii, D_ii_gpu)
