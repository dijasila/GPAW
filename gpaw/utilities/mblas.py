import numpy as np
import sys

from gpaw.utilities import is_contiguous
from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import scal
from gpaw import debug
from gpaw import gpuarray

import _gpaw
import gpaw.gpu


def multi_axpy_cpu(a, x, y):
    """
    """
    for ai, xi, yi in zip(a, x, y):
        axpy(ai, xi, yi)

def multi_axpy(a, x, y):
    """
    """
    assert type(x) == type(y)

    if isinstance(a, (float, complex)):
        axpy(a, x, y)
    else:
        if isinstance(x, gpuarray.GPUArray):
            if gpaw.gpu.debug:
                y_cpu = y.get()
                if isinstance(a, gpuarray.GPUArray):
                    multi_axpy_cpu(a.get(), x.get(), y_cpu)
                else:
                    multi_axpy_cpu(a, x.get(), y_cpu)

            if isinstance(a, gpuarray.GPUArray):
                _gpaw.multi_axpy_cuda_gpu(a.gpudata, a.dtype, x.gpudata,
                                          x.shape, y.gpudata, y.shape,
                                          x.dtype)
            else:
                a_gpu = gpuarray.to_gpu(a)
                _gpaw.multi_axpy_cuda_gpu(a_gpu.gpudata, a.dtype,
                                          x.gpudata, x.shape,
                                          y.gpudata, y.shape, x.dtype)
                if gpaw.gpu.debug:
                    gpaw.gpu.debug_test(y, y_cpu, "multi_axpy",
                                         raise_error=True)
        else:
            multi_axpy_cpu(a, x, y)

# Multivector dot product, a^H b, where ^H is transpose
def multi_dotc_cpu(x ,y, s):
    """
    """
    for i in range(len(s)):
        s[i] = dotc(x[i], y[i])

def multi_dotc(x, y, s=None):
    """
    """
    assert type(x) == type(y)

    if len(x.shape) == 1:
        return dotc(x, y)

    if isinstance(x, gpuarray.GPUArray):
        if isinstance(s, gpuarray.GPUArray):
            s_gpu = s
        else:
            s_gpu = gpuarray.empty(x.shape[0], dtype=x.dtype)
        _gpaw.multi_dotc_cuda_gpu(x.gpudata, x.shape, y.gpudata, x.dtype,
                                  s_gpu.gpudata)
        if gpaw.gpu.debug:
            s_cpu = np.empty(x.shape[0], dtype=x.dtype)
            multi_dotc_cpu(x.get(), y.get(), s_cpu)
            gpaw.gpu.debug_test(s_gpu, s_cpu, "multi_dotc")
        if not isinstance(s, gpuarray.GPUArray):
            s = s_gpu.get(s)
    else:
        if s is None:
            s = np.empty(x.shape[0], dtype=x.dtype)
        multi_dotc_cpu(x, y, s)
    return s

def multi_dotu_cpu(x, y, s):
    """
    """
    for i in range(len(s)):
        s[i] = dotu(x[i], y[i])

def multi_dotu(x, y, s=None):
    """
    """
    assert type(x) == type(y)

    if len(x.shape) == 1:
        return dotu(x, y)

    if isinstance(x, gpuarray.GPUArray):
        if isinstance(s, gpuarray.GPUArray):
            s_gpu = s
        else:
            s_gpu = gpuarray.empty(x.shape[0], dtype=x.dtype)
        _gpaw.multi_dotu_cuda_gpu(x.gpudata, x.shape, y.gpudata, x.dtype,
                                  s_gpu.gpudata)
        if gpaw.gpu.debug:
            s_cpu = np.empty(x.shape[0], dtype=x.dtype)
            multi_dotu_cpu(x.get(), y.get(), s_cpu)
            gpaw.gpu.debug_test(s_gpu, s_cpu, "multi_dotu")
        if not isinstance(s, gpuarray.GPUArray):
            s = s_gpu.get(s)
    else:
        if s is None:
            s = np.empty(x.shape[0], dtype=x.dtype)
        multi_dotu_cpu(x, y, s)
    return s

def multi_scal_cpu(a, x):
    """
    """
    for ai, xi in zip(a, x):
        scal(ai, xi)

def multi_scal(a, x):
    """
    """
    if isinstance(a, (float, complex)):
        scal(a, x)
    else:
        if isinstance(x, gpuarray.GPUArray):
            if gpaw.gpu.debug:
                x_cpu = x.get()
                if isinstance(a, gpuarray.GPUArray):
                    multi_scal_cpu(a.get(), x_cpu)
                else:
                    multi_scal_cpu(a, x_cpu)
            if isinstance(a, gpuarray.GPUArray):
                _gpaw.multi_scal_cuda_gpu(a.gpudata, a.dtype, x.gpudata,
                                          x.shape, x.dtype)
            else:
                a_gpu = gpuarray.to_gpu(a)
                _gpaw.multi_scal_cuda_gpu(a_gpu.gpudata, a.dtype,
                                          x.gpudata, x.shape, x.dtype)
            if gpaw.gpu.debug:
                gpaw.gpu.debug_test(x, x_cpu, "multi_scal")
        else:
            multi_scal_cpu(a, x)



