import pytest
import numpy as np
from gpaw.utilities.blas import axpy, mmm, rk, r2k

def test_blas(gpu):

    rng = np.random.default_rng(seed=42)
    for dtype in (float, complex):
        a = np.zeros((100, 100), dtype=dtype)
        b = np.zeros_like(a)
        if dtype == float:
            a[:] = rng.random((100, 100))
            b[:] = rng.random((100, 100))
        else:
            a.real = rng.random((100, 100))
            a.imag = rng.random((100, 100))
            b.real = rng.random((100, 100))
            b.imag = rng.random((100, 100))
            
        
        c = np.zeros_like(a)

        a_gpu = gpu.copy_to_device(a)
        b_gpu = gpu.copy_to_device(b)
        c_gpu = gpu.array.zeros_like(a_gpu)

        # axpy
        axpy(0.5, a, c)
        check_cpu = c.sum()
    
        axpy(0.5, a_gpu, c_gpu, cuda=True)
        check_gpu = gpu.copy_to_host(c_gpu.sum())

        assert check_cpu == pytest.approx(check_gpu, abs=1e-14) 

        # mmm
        mmm(0.5, a, 'N', b, 'N', 0.2, c)
        check_cpu = c.sum()

        mmm(0.5, a_gpu, 'N', b_gpu, 'N', 0.2, c_gpu, cuda=True)
        check_gpu = gpu.copy_to_host(c_gpu.sum())

        assert check_cpu == pytest.approx(check_gpu, rel=1e-14) 

        # rk
        rk(0.5, a, 0.2, c)
        check_cpu = c.sum()
    
        rk(0.5, a_gpu, 0.2, c_gpu, cuda=True)
        check_gpu = gpu.copy_to_host(c_gpu.sum())

        assert check_cpu == pytest.approx(check_gpu, rel=1e-14) 
 
        # r2k
        r2k(0.5, a, b, 0.2, c)
        check_cpu = c.sum()
    
        r2k(0.5, a_gpu, b_gpu, 0.2, c_gpu, cuda=True)
        check_gpu = gpu.copy_to_host(c_gpu.sum())

        assert check_cpu == pytest.approx(check_gpu, rel=1e-14) 
