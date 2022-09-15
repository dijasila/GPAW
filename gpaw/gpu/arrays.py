class BaseArrayInterface:
    _module = None  # replace with array module (pycuda, numpy, ...)
    _array = None   # replace with array class (GPUArray, numpy.array, ...)

    @property
    def Array(self):
        return self._array

    @Array.setter
    def Array(self, x):
        self._array = x

    def axpbyz(self, a, x, b, y, z):
        return self._module.axpbyz(a, x, b, y, z)

    def axpbz(self, a, x, b, z):
        return self._module.axpbz(a, x, b, z)

    def sum(self, x, axis=0, out=None):
        return self._module.sum(x, axis=axis, out=out)

    def to_gpu(self, x):
        return self._module.to_gpu(x)

    def to_gpu_async(self, x, stream=None):
        return self._module.to_gpu_async(x, stream=stream)

    def empty(self, shape, dtype=float, order='C'):
        return self._module.empty(shape, dtype=dtype, order=order)

    def zeros(self, shape, dtype=float, order='C'):
        return self._module.zeros(shape, dtype=dtype, order=order)

    def empty_like(self, x):
        return self._module.empty_like(x)

    def zeros_like(self, x):
        return self._module.zeros_like(x)

    def get_pointer(self, x):
        return self._module.get_pointer(x)

    def get_slice(self, x, shape):
        return self._module.get_slice(x, shape)


class PyCudaArrayInterface(BaseArrayInterface):
    from gpaw.gpu import pycuda as _module
    _array = _module.GPUArray

    def sum(self, x, axis=0, out=None):
        return self._module.sum(x, axis=axis, result=out)


class HostArrayInterface(BaseArrayInterface):
    import numpy as _module
    _array = _module.array

    def axpbyz(self, a, x, b, y, z):
        z = a * x + b * y

    def axpbz(self, a, x, b, z):
        z = a * x + b

    def to_gpu(self, x):
        return x.copy()

    def to_gpu_async(self, x, stream=None):
        return x.copy()

    def get_pointer(self, x):
        return x.ctypes.data

    def get_slice(self, x, shape):
        slices = [slice(0,x) for x in shape]
        return x[slices]
