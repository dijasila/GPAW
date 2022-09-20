import numpy as np
import sys

import _gpaw
from gpaw.utilities import is_contiguous
from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import scal
from gpaw import debug
from gpaw import gpu


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
        if gpu.is_device_array(x):
            if gpu.debug:
                y_cpu = gpu.copy_to_host(y)
                if gpu.is_device_array(a):
                    multi_axpy_cpu(gpu.copy_to_host(a),
                                   gpu.copy_to_host(x),
                                   y_cpu)
                else:
                    multi_axpy_cpu(a, gpu.copy_to_host(x), y_cpu)

            if gpu.is_device_array(a):
                _gpaw.multi_axpy_cuda_gpu(gpu.array.get_pointer(a),
                                          a.dtype,
                                          gpu.array.get_pointer(x),
                                          x.shape,
                                          gpu.array.get_pointer(y),
                                          y.shape,
                                          x.dtype)
            else:
                a_gpu = gpu.copy_to_device(a)
                _gpaw.multi_axpy_cuda_gpu(gpu.array.get_pointer(a_gpu),
                                          a.dtype,
                                          gpu.array.get_pointer(x),
                                          x.shape,
                                          gpu.array.get_pointer(y),
                                          y.shape,
                                          x.dtype)
                if gpu.debug:
                    gpu.debug_test(y, y_cpu, "multi_axpy", raise_error=True)
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

    if gpu.is_device_array(x):
        if gpu.is_device_array(s):
            s_gpu = s
        else:
            s_gpu = gpu.array.empty(x.shape[0], dtype=x.dtype)
        _gpaw.multi_dotc_cuda_gpu(gpu.array.get_pointer(x),
                                  x.shape,
                                  gpu.array.get_pointer(y),
                                  x.dtype,
                                  gpu.array.get_pointer(s_gpu))
        if gpu.debug:
            s_cpu = np.empty(x.shape[0], dtype=x.dtype)
            multi_dotc_cpu(gpu.copy_to_host(x), gpu.copy_to_host(y), s_cpu)
            gpu.debug_test(s_gpu, s_cpu, "multi_dotc")
        if gpu.is_host_array(s):
            s = gpu.copy_to_host(s_gpu, s)
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

    if gpu.is_device_array(x):
        if gpu.is_device_array(s):
            s_gpu = s
        else:
            s_gpu = gpu.array.empty(x.shape[0], dtype=x.dtype)
        _gpaw.multi_dotu_cuda_gpu(gpu.array.get_pointer(x),
                                  x.shape,
                                  gpu.array.get_pointer(y),
                                  x.dtype,
                                  gpu.array.get_pointer(s_gpu))
        if gpu.debug:
            s_cpu = np.empty(x.shape[0], dtype=x.dtype)
            multi_dotu_cpu(gpu.copy_to_host(x), gpu.copy_to_host(y), s_cpu)
            gpu.debug_test(s_gpu, s_cpu, "multi_dotu")
        if gpu.is_host_array(s):
            s = gpu.copy_to_host(s_gpu, s)
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
        if gpu.is_device_array(x):
            if gpu.debug:
                x_cpu = gpu.copy_to_host(x)
                if gpu.is_device_array(a):
                    multi_scal_cpu(gpu.copy_to_host(a), x_cpu)
                else:
                    multi_scal_cpu(a, x_cpu)
            if gpu.is_device_array(a):
                _gpaw.multi_scal_cuda_gpu(gpu.array.get_pointer(a),
                                          a.dtype,
                                          gpu.array.get_pointer(x),
                                          x.shape,
                                          x.dtype)
            else:
                a_gpu = gpu.copy_to_device(a)
                _gpaw.multi_scal_cuda_gpu(gpu.array.get_pointer(a_gpu),
                                          a.dtype,
                                          gpu.array.get_pointer(x),
                                          x.shape,
                                          x.dtype)
            if gpu.debug:
                gpu.debug_test(x, x_cpu, "multi_scal")
        else:
            multi_scal_cpu(a, x)
