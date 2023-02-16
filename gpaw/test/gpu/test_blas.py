import pytest
import numpy as np
from gpaw.utilities.blas import (axpy, dotc, dotu, gemm, gemv, mmm, rk, r2k,
                                 scal)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [float, complex])
def test_blas(gpu, dtype):
    rng = np.random.default_rng(seed=42)
    a = np.zeros((100, 100), dtype=dtype)
    b = np.zeros_like(a)
    x = np.zeros((100,), dtype=dtype)
    y = np.zeros_like(x)
    if dtype == float:
        a[:] = rng.random((100, 100))
        b[:] = rng.random((100, 100))
        x[:] = rng.random((100,))
        y[:] = rng.random((100,))
    else:
        a.real = rng.random((100, 100))
        a.imag = rng.random((100, 100))
        b.real = rng.random((100, 100))
        b.imag = rng.random((100, 100))
        x.real = rng.random((100,))
        x.imag = rng.random((100,))
        y.real = rng.random((100,))
        y.imag = rng.random((100,))

    c = np.zeros_like(a)

    a_gpu = gpu.copy_to_device(a)
    b_gpu = gpu.copy_to_device(b)
    c_gpu = gpu.cupy.zeros_like(a_gpu)
    x_gpu = gpu.copy_to_device(x)
    y_gpu = gpu.copy_to_device(y)

    # axpy
    axpy(0.5, a, c)
    check_cpu = c.sum()

    axpy(0.5, a_gpu, c_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(c_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, abs=1e-14)

    # mmm
    mmm(0.5, a, 'N', b, 'N', 0.2, c)
    check_cpu = c.sum()

    mmm(0.5, a_gpu, 'N', b_gpu, 'N', 0.2, c_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(c_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # gemm
    gemm(0.5, a, b, 0.2, c)
    check_cpu = c.sum()

    gemm(0.5, a_gpu, b_gpu, 0.2, c_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(c_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # gemv
    gemv(0.5, a, x, 0.2, c)
    check_cpu = c.sum()

    gemv(0.5, a_gpu, x_gpu, 0.2, c_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(c_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # rk
    rk(0.5, a, 0.2, c)
    check_cpu = c.sum()

    rk(0.5, a_gpu, 0.2, c_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(c_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # r2k
    r2k(0.5, a, b, 0.2, c)
    check_cpu = c.sum()

    r2k(0.5, a_gpu, b_gpu, 0.2, c_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(c_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # dotc
    check_cpu = dotc(x, y)

    check_gpu = dotc(x_gpu, y_gpu)

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # dotu
    check_cpu = dotu(x, y)

    check_gpu = dotu(x_gpu, y_gpu)

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

    # scal
    scal(0.5, a)
    check_cpu = a.sum()

    scal(0.5, a_gpu)
    check_gpu = gpu.copy_to_host(a_gpu.sum())

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)
