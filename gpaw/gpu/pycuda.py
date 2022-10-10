import numpy as np

import pycuda.driver as drv
from pycuda.gpuarray import GPUArray as pycuda_GPUArray

import _gpaw


class GPUArray(pycuda_GPUArray):
    """GPUArray that supports slicing in the 1st dimension.

    A slightly modified version of PyCUDA's GPUArray to support slicing
    in the first dimension. Original code from PyCUDA is mostly untouched
    and methods reshape() and view() are only modified to return an
    instance of this class.

    Uses our own CUDA kernels for axpbyz(), axpbz(), and fill().

    Ideally, once slicing is supported by PyCUDA's GPUArray we could
    simply use it directly and drop this subclass."""

    def reshape(self, *shape, **kwargs):
        order = kwargs.pop("order", "C")
        x = super().reshape(*shape, **kwargs)
        return self.__class__(
                shape=x.shape,
                dtype=self.dtype,
                allocator=self.allocator,
                base=self,
                gpudata=int(self.gpudata),
                order=order)

    def view(self, dtype=None):
        x = super().view(dtype)
        return self.__class__(
                shape=x.shape,
                dtype=x.dtype,
                allocator=self.allocator,
                strides=x.strides,
                base=self,
                gpudata=int(self.gpudata))

    def __getitem__(self, index):
        x = super().__getitem__(index)
        return self.__class__(
                shape=x.shape,
                dtype=self.dtype,
                allocator=self.allocator,
                strides=x.strides,
                base=self,
                gpudata=int(x.gpudata))

    def _axpbyz(self, selffac, other, otherfac, out, add_timer=None, stream=None):
        """Compute ``out = selffac * self + otherfac*other``,
        where `other` is a vector.."""
        assert self.shape == other.shape
        if not self.flags.forc or not other.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                    "be used as arguments to this operation")
        axpbyz(selffac, self, otherfac, other, out)
        return out

    def _axpbz(self, selffac, other, out, stream=None):
        """Compute ``out = selffac * self + other``, where `other` is a scalar."""
        if not self.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                    "be used as arguments to this operation")
        axpbz(selffac, self, other, out)
        return out

    def fill(self, value, stream=None):
        """fills the array with the specified value"""
        _gpaw.fill_gpu(value, self.gpudata, self.shape, self.dtype)
        return self

def axpbyz(a, x, b, y, z):
    _gpaw.axpbyz_gpu(a, x.gpudata, b, y.gpudata, z.gpudata, x.shape, x.dtype)
    return z

def axpbz(a, x, b, z):
    _gpaw.axpbz_gpu(a, x.gpudata, b, z.gpudata, x.shape, x.dtype)
    return z

def sum(x, result=None, axis=0):
    """sum of array elements over a given axis"""
    if not isinstance(x, GPUArray) and not isinstance(result, GPUArray):
        return np.sum(x, axis=axis, out=result)
    if axis > 0 or not isinstance(x, GPUArray):
        if isinstance(x, GPUArray):
            x = x.get()
        if isinstance(result, GPUArray):
            result.set(np.sum(x, axis=axis))
        else:
            result = np.sum(x, axis=axis, out=result)
        return result
    shape = x.shape[:axis] + x.shape[axis+1:]
    convert = False
    if result is None:
        result = GPUArray(shape, x.dtype)
    elif not isinstance(result, GPUArray):
        _result = result
        result = GPUArray(shape, x.dtype)
        convert = True
    ones = empty_like(result)
    ones.fill(1.0)
    _gpaw.gemv_cuda_gpu(1.0, x.gpudata, x.shape, ones.gpudata, ones.shape,
                        1.0, result.gpudata, x.dtype, 'n')
    if convert:
        result.get(_result)
        return _result
    return result

def to_gpu(ary, allocator=drv.mem_alloc):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, allocator, strides=ary.strides)
    result.set(ary)
    return result

def to_gpu_async(ary, allocator=drv.mem_alloc, stream=None):
    """converts a numpy array to a GPUArray"""
    result = GPUArray(ary.shape, ary.dtype, allocator, strides=ary.strides)
    result.set_async(ary, stream)
    return result

empty = GPUArray

def zeros(shape, dtype, allocator=drv.mem_alloc, order="C"):
    """Returns an array of the given shape and dtype filled with 0's."""

    result = GPUArray(shape, dtype, allocator, order=order)
    result.fill(0.0)
    return result

def empty_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, other_ary.allocator)
    return result

def zeros_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, other_ary.allocator)
    result.fill(0.0)
    return result

def get_pointer(ary):
    return ary.gpudata

def get_slice(ary, shape):
    return GPUArray(base=ary, allocator=ary.allocator, gpudata=ary.gpudata,
                    dtype=ary.dtype, shape=shape)
