import pytest
import numpy as np

def test_array(gpu):

    rng = np.random.RandomState(42)
    a = rng.random((100, 100))
    b = rng.random((100, 100))
    c = np.zeros_like(a)

    c[:] = a + b
    sum_cpu = c.sum()

    a_gpu = gpu.copy_to_device(a)
    b_gpu = gpu.copy_to_device(b)
    c_gpu = gpu.array.zeros_like(a_gpu)

    c_gpu[:] = a_gpu + b_gpu
    sum_gpu = gpu.copy_to_host(c_gpu.sum())

    assert sum_cpu == pytest.approx(sum_gpu, abs=1e-14) 
