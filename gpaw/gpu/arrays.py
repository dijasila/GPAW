class BaseArrayInterface:
    def __init__(self):
        self._module = None  # replace with array module (cupy, numpy, ...)
        self.Array = None    # replace with array class (numpy.ndarray ...)

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


class CuPyArrayInterface(BaseArrayInterface):
    def __init__(self):
        import cupy as _module
        self._module = _module
        self.Array = _module.ndarray

    def axpbyz(self, a, x, b, y, z):
        z[:] = a * x + b * y

    def axpbz(self, a, x, b, z):
        z[:] = a * x + b

    def copy_to_host(self, src, tgt=None, stream=None):
        if stream is None:
            if tgt is not None:
                tgt[:] = self._module.asnumpy(src, stream)
            else:
                tgt = self._module.asnumpy(src, stream)
            self._module.cuda.runtime.deviceSynchronize()
        else:
            with stream:
                if tgt is not None:
                    tgt[:] = self._module.asnumpy(src, stream)
                else:
                    tgt = self._module.asnumpy(src, stream)
        return tgt

    def copy_to_device(self, src, tgt=None, stream=None):
        if stream is None:
            if tgt is not None:
                tgt[:] = self._module.asarray(src)
            else:
                tgt = self._module.asarray(src)
            self._module.cuda.runtime.deviceSynchronize()
        else:
            with stream:
                if tgt is not None:
                    tgt[:] = self._module.asarray(src)
                else:
                    tgt = self._module.asarray(src)
        return tgt

    def memcpy_dtod(self, tgt, src):
        tgt[:] = src[:]
        self._module.cuda.runtime.deviceSynchronize()

    def get_pointer(self, x):
        return x.data.ptr

    def get_slice(self, x, shape):
        slices = tuple([slice(n) for n in shape])
        return x[slices]


class HostArrayInterface(BaseArrayInterface):
    def __init__(self):
        import numpy as _module
        self._module = _module
        self.Array = _module.ndarray

    def axpbyz(self, a, x, b, y, z):
        z[:] = a * x + b * y

    def axpbz(self, a, x, b, z):
        z[:] = a * x + b

    def copy_to_host(self, src, tgt=None, stream=None):
        if tgt is not None:
            tgt[:] = src[:]
        else:
            tgt = src.copy()
        return tgt

    def copy_to_device(self, src, tgt=None, stream=None):
        return self.copy_to_host(src, tgt)

    def get_pointer(self, x):
        return x.ctypes.data

    def get_slice(self, x, shape):
        slices = [slice(n) for n in shape]
        return x[slices]
