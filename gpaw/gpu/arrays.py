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

    def copy_to_host(self, src, tgt=None, stream=None):
        return self._module.copy_to_host(src, tgt=tgt, stream=stream)

    def copy_to_device(self, src, tgt=None, stream=None):
        return self._module.copy_to_device(src, tgt=tgt, stream=stream)

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

    def copy_to_host(self, src, tgt=None, stream=None):
        if stream:
            return src.get_async(ary=tgt, stream=stream)
        else:
            return src.get(ary=tgt)

    def copy_to_device(self, src, tgt=None, stream=None):
        if stream:
            if tgt:
                tgt.set_async(src, stream=stream)
                return tgt
            else:
                return self._module.to_gpu_async(src, stream=stream)
        else:
            if tgt:
                tgt.set(src)
                return tgt
            else:
                return self._module.to_gpu(src)


class HostArrayInterface(BaseArrayInterface):
    import numpy as _module
    _array = _module.array

    def axpbyz(self, a, x, b, y, z):
        z = a * x + b * y

    def axpbz(self, a, x, b, z):
        z = a * x + b

    def copy_to_host(self, src, tgt=None, stream=None):
        if tgt:
            tgt[:] = src[:]
        else:
            tgt = src.copy()
        return tgt

    def copy_to_device(self, src, tgt=None, stream=None):
        return self.copy_to_host(src, tgt)

    def get_pointer(self, x):
        return x.ctypes.data

    def get_slice(self, x, shape):
        slices = [slice(0,x) for x in shape]
        return x[slices]
