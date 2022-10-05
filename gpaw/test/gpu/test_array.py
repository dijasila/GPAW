import pytest
import numpy as np
from gpaw import gpu

@pytest.mark.gpu
def test_array():
    gpu.setup(cuda=True)
    gpu.init()

    a = np.random.random((100, 100))
    b = np.random.random((100, 100))
    c = np.zeros_like(a)

    c[:] = a + b
    sum_cpu = c.sum()

    a_gpu = gpu.copy_to_device(a)
    b_gpu = gpu.copy_to_device(b)
    c_gpu = gpu.array.zeros_like(a_gpu)

    c_gpu[:] = a_gpu + b_gpu

    sum_gpu = gpu.copy_to_host(c_gpu.sum())

    assert sum_cpu == pytest.approx(sum_gpu, abs=1e-10) 
