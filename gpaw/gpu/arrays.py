class BaseArrayInterface:
    _module = None  # replace with array module (pycuda, numpy, ...)
    Array = None    # replace with array class (GPUArray, ndarray, ...)

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
    Array = _module.GPUArray

    def sum(self, x, axis=0, out=None):
        return self._module.sum(x, axis=axis, result=out)

    def copy_to_host(self, src, tgt=None, stream=None):
        if stream:
            return src.get_async(ary=tgt, stream=stream)
        else:
            return src.get(ary=tgt)

    def copy_to_device(self, src, tgt=None, stream=None):
        if stream:
            if tgt is not None:
                tgt.set_async(src, stream=stream)
                return tgt
            else:
                return self._module.to_gpu_async(src, stream=stream)
        else:
            if tgt is not None:
                tgt.set(src)
                return tgt
            else:
                return self._module.to_gpu(src)

class CuPyArrayInterface(BaseArrayInterface):
    import cupy as _module
    Array = _module.ndarray

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
            self._module.cuda.get_current_stream().synchronize()
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
            self._module.cuda.get_current_stream().synchronize()
        else:
            with stream:
                if tgt is not None:
                    tgt[:] = self._module.asarray(src)
                else:
                    tgt = self._module.asarray(src)
        return tgt

    def memcpy_dtod(self, tgt, src):
        tgt[:] = src[:]
        self._module.cuda.get_current_stream().synchronize()

    def get_pointer(self, x):
        return x.data.ptr

    def get_slice(self, x, shape):
        slices = tuple([slice(n) for n in shape])
        return x[slices]

class HostArrayInterface(BaseArrayInterface):
    import numpy as _module
    Array = _module.ndarray

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
