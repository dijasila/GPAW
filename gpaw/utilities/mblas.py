import numpy as np
import sys

import _gpaw
from gpaw.utilities import is_contiguous
from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import scal
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
        if not isinstance(x, np.ndarray):
            if not isinstance(a, np.ndarray):
                a_gpu = a
            else:
                a_gpu = gpu.copy_to_device(a)
            _gpaw.multi_axpy_gpu(gpu.get_pointer(a_gpu),
                                 a.dtype,
                                 gpu.get_pointer(x),
                                 x.shape,
                                 gpu.get_pointer(y),
                                 y.shape,
                                 x.dtype)
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

    if not isinstance(x, np.ndarray):
        if not isinstance(s, np.ndarray):
            s_gpu = s
        else:
            s_gpu = gpu.cupy.empty(x.shape[0], dtype=x.dtype)
        _gpaw.multi_dotc_gpu(gpu.get_pointer(x),
                             x.shape,
                             gpu.get_pointer(y),
                             x.dtype,
                             gpu.get_pointer(s_gpu))
        if isinstance(s, np.ndarray):
            s = gpu.copy_to_host(s_gpu, out=s)
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

    if not isinstance(x, np.ndarray):
        if not isinstance(s, np.ndarray):
            s_gpu = s
        else:
            s_gpu = gpu.cupy.empty(x.shape[0], dtype=x.dtype)
        _gpaw.multi_dotu_gpu(gpu.get_pointer(x),
                             x.shape,
                             gpu.get_pointer(y),
                             x.dtype,
                             gpu.get_pointer(s_gpu))
        if isinstance(s, np.ndarray):
            s = gpu.copy_to_host(s_gpu, out=s)
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
        if not isinstance(x, np.ndarray):
            if not isinstance(a, np.ndarray):
                a_gpu = a
            else:
                a_gpu = gpu.copy_to_device(a)
            _gpaw.multi_scal_gpu(gpu.get_pointer(a_gpu),
                                 a.dtype,
                                 gpu.get_pointer(x),
                                 x.shape,
                                 x.dtype)
        else:
            multi_scal_cpu(a, x)
