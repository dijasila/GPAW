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

    Ideally, once slicing is supported by PyCUDA's GPUArray we could
    simply use it directly and drop this subclass."""

    def reshape(self, *shape):
        # TODO: add more error-checking, perhaps
        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])
        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        return self.__class__(
                shape=shape,
                dtype=self.dtype,
                allocator=self.allocator,
                base=self,
                gpudata=int(self.gpudata))

    def view(self, dtype=None):
        if dtype is None:
            dtype = self.dtype

        old_itemsize = self.dtype.itemsize
        itemsize = np.dtype(dtype).itemsize

        if self.shape[-1] * old_itemsize % itemsize != 0:
            raise ValueError("new type not compatible with array")

        shape = self.shape[:-1] \
              + (self.shape[-1] * old_itemsize // itemsize,)

        return self.__class__(
                shape=shape,
                dtype=dtype,
                allocator=self.allocator,
                base=self,
                gpudata=int(self.gpudata))

    # slicing
    def __getitem__(self, idx):
        if idx == ():
            return self

        if len(self.shape) > 1:
            if isinstance(idx, int):
                if idx >= self.shape[0]:
                    raise IndexError("index out of bounds")
                return self.__class__(
                    shape=self.shape[1:],
                    dtype=self.dtype,
                    allocator=self.allocator,
                    base=self,
                    gpudata=int(self.gpudata)
                           + self.dtype.itemsize * idx * self.size
                           // len(self))
            elif isinstance(idx, slice):
                start, stop, stride = idx.indices(len(self))
                return self.__class__(
                    shape=((stop - start) // stride,) + self.shape[1:],
                    dtype=self.dtype,
                    allocator=self.allocator,
                    base=self,
                    gpudata=int(self.gpudata)
                           + self.dtype.itemsize * start * self.size
                           // len(self))
            raise NotImplementedError("multi-d slicing is not yet implemented")

        if not isinstance(idx, slice):
            raise ValueError("non-slice indexing not supported: %s" \
                    % (idx,))

        l, = self.shape
        start, stop, stride = idx.indices(l)

        if stride != 1:
            raise NotImplementedError("strided slicing is not yet implemented")

        return self.__class__(
                shape=((stop - start) // stride,),
                dtype=self.dtype,
                allocator=self.allocator,
                base=self,
                gpudata=int(self.gpudata) + start * self.dtype.itemsize)

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
    zero = np.zeros((), dtype)
    result.fill(zero)
    return result

def empty_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, other_ary.allocator)
    return result

def zeros_like(other_ary):
    result = GPUArray(
            other_ary.shape, other_ary.dtype, other_ary.allocator)
    zero = np.zeros((), result.dtype)
    result.fill(zero)
    return result
