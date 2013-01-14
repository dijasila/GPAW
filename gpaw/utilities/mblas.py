import numpy as np
import sys

from gpaw.utilities import is_contiguous
from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import scal

from gpaw import debug

import _gpaw

import gpaw.cuda



def multi_axpy_cpu(a,x,y):
    """
    """
    for ai,xi,yi in zip(a,x,y):
        axpy(ai, xi, yi)

        
def multi_axpy(a,x,y):
    """
    """
    assert type(x) == type(y)
    #print "multi_axpy"
    if isinstance(a, (float, complex)):
        axpy(a, x, y)
    else:
        if isinstance(x,gpaw.cuda.gpuarray.GPUArray):                
            if gpaw.cuda.debug:
                y_cpu=y.get()
                if isinstance(a,gpaw.cuda.gpuarray.GPUArray):
                    multi_axpy_cpu(a.get(),x.get(),y_cpu)
                else:
                    multi_axpy_cpu(a,x.get(),y_cpu)

            if isinstance(a,gpaw.cuda.gpuarray.GPUArray):                
                _gpaw.multi_axpy_cuda_gpu(a.gpudata,a.dtype, x.gpudata, x.shape, y.gpudata, y.shape, x.dtype)
            else:
                _gpaw.multi_axpy_cuda_gpu(gpaw.cuda.gpuarray.to_gpu(a).gpudata,a.dtype, x.gpudata, x.shape, y.gpudata, y.shape, x.dtype)
            if gpaw.cuda.debug:
                gpaw.cuda.debug_test(y,y_cpu,"multi_axpy",raise_error=True)
        else:
            multi_axpy_cpu(a,x,y)


# Multivector dot product, a^H b, where ^H is transpose
def multi_dotc_cpu(x,y,s):
    """
    """
    for i in range(len(s)):
        s[i] = dotc(x[i],y[i])

    
def multi_dotc(x,y,s=None):
    """
    """

    assert type(x) == type(y)


    if len(x.shape)==1:
         return dotc(x,y)
    elif s is None:
        s=np.empty(x.shape[0],dtype=x.dtype)
        
    if isinstance(x,gpaw.cuda.gpuarray.GPUArray):
        _gpaw.multi_dotc_cuda_gpu(x.gpudata, x.shape, y.gpudata, x.dtype, s)
        if gpaw.cuda.debug:
            s_cpu=np.empty(x.shape[0],dtype=x.dtype)        
            multi_dotc_cpu(x.get(), y.get(),s_cpu)
            if not gpaw.cuda.debug_test(s,s_cpu,"multi_dotc"):
                print abs(s_cpu-s)
                assert(0)
    else:
        multi_dotc_cpu(x,y,s)

    return s


def multi_dotu_cpu(x,y,s):
    """
    """
    for i in range(len(s)):
        s[i] = dotu(x[i],y[i])


    
def multi_dotu(x,y,s=None):
    """
    """

    assert type(x) == type(y)

    if len(x.shape)==1:
         return dotu(x,y)
    elif s is None:
        s=np.empty(x.shape[0],dtype=x.dtype)
    
    if isinstance(x,gpaw.cuda.gpuarray.GPUArray):
        _gpaw.multi_dotu_cuda_gpu(x.gpudata, x.shape, y.gpudata, x.dtype, s)
        if gpaw.cuda.debug:
            s_cpu=np.empty(x.shape[0],dtype=x.dtype)
            multi_dotu_cpu(x.get(), y.get(),s_cpu)
            gpaw.cuda.debug_test(s,s_cpu,"multi_dotu")
    else:
        multi_dotu_cpu(x,y,s)
    return s

            
def multi_scal_cpu(a,x):
    """
    """
    for ai,xi in zip(a,x):
        scal(ai,xi)

            
def multi_scal(a,x):
    """
    """
    #print "multi_scal"    
    if isinstance(a, (float, complex)):
        scal(a,x)
    else:
        if isinstance(x,gpaw.cuda.gpuarray.GPUArray):
            if gpaw.cuda.debug:
                x_cpu=x.get()
                if isinstance(a,gpaw.cuda.gpuarray.GPUArray):                
                    multi_scal_cpu(a.get(),x_cpu)
                else:
                    multi_scal_cpu(a,x_cpu)
                    
            if isinstance(a,gpaw.cuda.gpuarray.GPUArray):                
                _gpaw.multi_scal_cuda_gpu(a.gpudata, a.dtype, x.gpudata, x.shape, x.dtype)
            else:
                _gpaw.multi_scal_cuda_gpu(gpaw.cuda.gpuarray.to_gpu(a).gpudata,a.dtype, x.gpudata, x.shape, x.dtype)
            if gpaw.cuda.debug:
                gpaw.cuda.debug_test(x,x_cpu,"multi_scal")
        else:
            multi_scal_cpu(a,x)



