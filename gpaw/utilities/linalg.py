import numpy as np
from gpaw import debug

import _gpaw

from gpaw.utilities.blas import axpy,scal

from gpaw.cuda import debug_cuda,debug_cuda_test
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


def elementwise_multiply_add(a,b,c):
    """
    """
    assert(type(a)==type(b))
    assert(type(c)==type(b))
    
    if isinstance(a,gpuarray.GPUArray):
        if debug_cuda:
            c_cpu=c.get()
        _gpaw.elementwise_multiply_add_gpu(a.gpudata,a.shape,a.dtype,
                                           b.gpudata,b.dtype,
                                           c.gpudata)
        if debug_cuda:
            c_cpu+=a.get()*b.get()
            debug_cuda_test(c.get(),c_cpu,"elementwise_multiply_add")
    else:
        c+=a*b


def multi_elementwise_multiply_add_cpu(a,b,c):

    if a.ndim>b.ndim:
        for ci, ai in zip(c,a):
            ci+=ai*b
    else:
        for ci, bi in zip(c,b):
            ci+=a*bi

def multi_elementwise_multiply_add(a,b,c):
    """
    """
    assert(type(a)==type(b))
    assert(type(c)==type(b))

    if len(a.shape) == len(b.shape):
        elementwise_multiply_add(a,b,c)
            
    if isinstance(a,gpuarray.GPUArray):
        if debug_cuda:
            c_cpu=c.get()
        _gpaw.multi_elementwise_multiply_add_gpu(a.gpudata,a.shape,a.dtype,
                                                 b.gpudata,b.shape,b.dtype,
                                                 c.gpudata)
        if debug_cuda:
            multi_elementwise_multiply_add_cpu(a.get(),b.get(),c_cpu)
            debug_cuda_test(c.get(),c_cpu,"multi_elementwise_multiply_add")
    else:
        multi_elementwise_multiply_add_cpu(a,b,c)




def change_sign(x):
    """
    """
    if isinstance(x,gpuarray.GPUArray):
        if debug_cuda:
            x_cpu=-x.get()
        _gpaw.csign_gpu(x.gpudata,x.shape,x.dtype)
        if debug_cuda:
            debug_cuda_test(x.get(),x_cpu,"neg")
        
    else:
        scal(-1.0,x)


def ax2py_cpu(a,x,y):
    if x.dtype == float:
        axpy(a, x*x, y)
    else:
        axpy(a, x.real*x.real, y)
        axpy(a, x.imag*x.imag, y)



def ax2py(a,x,y):
    """
    """
    assert(type(x)==type(y))
    if isinstance(x,gpuarray.GPUArray):
        if debug_cuda:
            y_cpu=y.get()
        _gpaw.ax2py_gpu(a,x.gpudata,x.shape,
                        y.gpudata,y.shape,
                        x.dtype)
        if debug_cuda:
            ax2py_cpu(a,x.get(),y_cpu)
            debug_cuda_test(y.get(),y_cpu,"ax2py")
        
    else:
        ax2py_cpu(a,x,y)



def multi_ax2py_cpu(a,x,y):
    """
    """
    for ai,xi in zip(a,x):
        ax2py_cpu(ai, xi, y)

        
def multi_ax2py(a,x,y):
    """
    """
    assert type(x) == type(y)

    if isinstance(a, (float, complex)):
        ax2py(a, x, y)
    else:
        if isinstance(x,gpuarray.GPUArray):                
            if debug_cuda:
                y_cpu=y.get()
                if isinstance(a,gpuarray.GPUArray):
                    multi_ax2py_cpu(a.get(),x.get(),y_cpu)
                else:
                    multi_ax2py_cpu(a,x.get(),y_cpu)

            if isinstance(a,gpuarray.GPUArray):                
                _gpaw.multi_ax2py_gpu(a.gpudata, x.gpudata, x.shape, y.gpudata, y.shape, x.dtype)
            else:
                _gpaw.multi_ax2py_gpu(gpuarray.to_gpu(a).gpudata, x.gpudata, x.shape, y.gpudata, y.shape, x.dtype)
            if debug_cuda:
                debug_cuda_test(y.get(),y_cpu,"multi_ax2py")
        else:
            multi_ax2py_cpu(a,x,y)
