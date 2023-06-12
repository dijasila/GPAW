import pytest
import numpy as np
from gpaw.utilities.blas import gpu_axpy, mmm
from gpaw.gpu import cupy as cp


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [float, complex])
def test_blas(gpu, dtype):
    N = 100
    rng = np.random.default_rng(seed=42)
    a = np.zeros((N, N), dtype=dtype)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    x = np.zeros((N,), dtype=dtype)
    y = np.zeros_like(x)
    if dtype == float:
        a[:] = rng.random((N, N))
        b[:] = rng.random((N, N))
        c[:] = rng.random((N, N))
        x[:] = rng.random((N,))
        y[:] = rng.random((N,))
    else:
        a.real = rng.random((N, N))
        a.imag = rng.random((N, N))
        b.real = rng.random((N, N))
        b.imag = rng.random((N, N))
        c.real = rng.random((N, N))
        c.imag = rng.random((N, N))
        x.real = rng.random((N,))
        x.imag = rng.random((N,))
        y.real = rng.random((N,))
        y.imag = rng.random((N,))

    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)

    # axpy
    y += 0.5 * x
    check_cpu = y.sum()

    gpu_axpy(0.5, x_gpu, y_gpu)
    check_gpu = y_gpu.sum().get()

    assert check_cpu == pytest.approx(check_gpu, rel=1e-14)

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
    gemv(0.5, a, x, 0.2, y)
    check_cpu = y.sum()

    gemv(0.5, a_gpu, x_gpu, 0.2, y_gpu, use_gpu=True)
    check_gpu = gpu.copy_to_host(y_gpu.sum())

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
