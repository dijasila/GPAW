import numpy as np

import _gpaw
from gpaw import debug
from gpaw.utilities.blas import axpy, scal
from gpaw import gpu


def elementwise_multiply_add(a, b, c):
    """
    """
    assert(type(a) == type(b))
    assert(type(c) == type(b))

    if gpu.is_device_array(a):
        _gpaw.elementwise_multiply_add_gpu(gpu.array.get_pointer(a),
                                           a.shape,
                                           a.dtype,
                                           gpu.array.get_pointer(b),
                                           b.dtype,
                                           gpu.array.get_pointer(c))
    else:
        c += a * b

def multi_elementwise_multiply_add_cpu(a, b, c):
    if a.ndim > b.ndim:
        for ci, ai in zip(c, a):
            ci += ai * b
    else:
        for ci, bi in zip(c, b):
            ci += a * bi

def multi_elementwise_multiply_add(a, b, c):
    """
    """
    assert(type(a) == type(b))
    assert(type(c) == type(b))

    if len(a.shape) == len(b.shape):
        elementwise_multiply_add(a, b, c)

    if gpu.is_device_array(a):
        _gpaw.multi_elementwise_multiply_add_gpu(gpu.array.get_pointer(a),
                                                 a.shape,
                                                 a.dtype,
                                                 gpu.array.get_pointer(b),
                                                 b.shape,
                                                 b.dtype,
                                                 gpu.array.get_pointer(c))
    else:
        multi_elementwise_multiply_add_cpu(a, b, c)

def change_sign(x):
    """
    """
    if gpu.is_device_array(x):
        _gpaw.csign_gpu(gpu.array.get_pointer(x), x.shape, x.dtype)
    else:
        scal(-1.0, x)

def ax2py_cpu(a, x, y):
    if x.dtype == float:
        axpy(a, x * x, y)
    else:
        axpy(a, x.real * x.real, y)
        axpy(a, x.imag * x.imag, y)

def ax2py(a, x, y):
    """
    """
    assert(type(x) == type(y))
    if gpu.is_device_array(x):
        _gpaw.ax2py_gpu(a, gpu.array.get_pointer(x), x.shape,
                        gpu.array.get_pointer(y), y.shape, x.dtype)
    else:
        ax2py_cpu(a, x, y)

def multi_ax2py_cpu(a, x, y):
    """
    """
    for ai, xi in zip(a, x):
        ax2py_cpu(ai, xi, y)

def multi_ax2py(a, x, y):
    """
    """
    assert type(x) == type(y)

    if isinstance(a, (float, complex)):
        ax2py(a, x, y)
    else:
        if gpu.is_device_array(x):
            if gpu.is_device_array(a):
                a_gpu = a
            else:
                a_gpu = gpu.copy_to_device(a)
            _gpaw.multi_ax2py_gpu(gpu.array.get_pointer(a_gpu),
                                  gpu.array.get_pointer(x),
                                  x.shape,
                                  gpu.array.get_pointer(y),
                                  y.shape,
                                  x.dtype)
        else:
            multi_ax2py_cpu(a, x, y)
