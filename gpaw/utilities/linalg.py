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
        if gpu.debug:
            c_cpu = gpu.copy_to_host(c)
        _gpaw.elementwise_multiply_add_gpu(gpu.array.get_pointer(a),
                                           a.shape,
                                           a.dtype,
                                           gpu.array.get_pointer(b),
                                           b.dtype,
                                           gpu.array.get_pointer(c))
        if gpu.debug:
            c_cpu += gpu.copy_to_host(a) * gpu.copy_to_host(b)
            gpu.debug_test(c, c_cpu, "elementwise_multiply_add")
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
        if gpu.debug:
            c_cpu = gpu.copy_to_host(c)
        _gpaw.multi_elementwise_multiply_add_gpu(gpu.array.get_pointer(a),
                                                 a.shape,
                                                 a.dtype,
                                                 gpu.array.get_pointer(b),
                                                 b.shape,
                                                 b.dtype,
                                                 gpu.array.get_pointer(c))
        if gpu.debug:
            multi_elementwise_multiply_add_cpu(gpu.copy_to_host(a),
                                               gpu.copy_to_host(b),
                                               c_cpu)
            gpu.debug_test(c, c_cpu, "multi_elementwise_multiply_add")
    else:
        multi_elementwise_multiply_add_cpu(a, b, c)

def change_sign(x):
    """
    """
    if gpu.is_device_array(x):
        if gpu.debug:
            x_cpu =- gpu.copy_to_host(x)
        _gpaw.csign_gpu(gpu.array.get_pointer(x), x.shape, x.dtype)
        if gpu.debug:
            gpu.debug_test(x, x_cpu, "neg")
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
        if gpu.debug:
            y_cpu = gpu.copy_to_host(y)
        _gpaw.ax2py_gpu(a, gpu.array.get_pointer(x), x.shape,
                        gpu.array.get_pointer(y), y.shape, x.dtype)
        if gpu.debug:
            ax2py_cpu(a, gpu.copy_to_host(x), y_cpu)
            gpu.debug_test(y, y_cpu, "ax2py")
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
            if gpu.debug:
                y_cpu = gpu.copy_to_host(y)
                if gpu.is_device_array(a):
                    multi_ax2py_cpu(gpu.copy_to_host(a),
                                    gpu.copy_to_host(x),
                                    y_cpu)
                else:
                    multi_ax2py_cpu(a, gpu.copy_to_host(x), y_cpu)

            if gpu.is_device_array(a):
                _gpaw.multi_ax2py_gpu(gpu.array.get_pointer(a),
                                      gpu.array.get_pointer(x),
                                      x.shape,
                                      gpu.array.get_pointer(y),
                                      y.shape,
                                      x.dtype)
            else:
                a_gpu = gpu.copy_to_device(a)
                _gpaw.multi_ax2py_gpu(gpu.array.get_pointer(a_gpu),
                                      gpu.array.get_pointer(x),
                                      x.shape,
                                      gpu.array.get_pointer(y),
                                      y.shape,
                                      x.dtype)
            if gpu.debug:
                gpu.debug_test(y, y_cpu, "multi_ax2py")
        else:
            multi_ax2py_cpu(a, x, y)
