import numpy as np
import gpaw.cpupy.linalg as linalg
import gpaw.cpupy.cublas as cublas

__all__ = ['cublas', 'linalg']


def empty(*args, **kwargs):
    return CuPyArray(np.empty(*args, **kwargs))


def asnumpy(a, out=None):
    if out is None:
        return a._data.copy()
    out[:] = a._data
    return out


def asarray(a):
    return CuPyArray(a.copy())


def multiply(a, b, c):
    np.multiply(a._data, b._data, c._data)


def negative(a, b):
    np.negative(a._data, b._data)


class CuPyArray:
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.size = data.size
        self.flags = data.flags

    @property
    def T(self):
        return CuPyArray(self._data.T)

    @property
    def imag(self):
        return CuPyArray(self._data.imag)

    def __iter__(self):
        for data in self._data:
            yield CuPyArray(data)

    def __setitem__(self, index, value):
        if isinstance(index, CuPyArray):
            index = index._data
        if isinstance(value, CuPyArray):
            self._data[index] = value._data
        else:
            assert isinstance(value, float)
            self._data[index] = value

    def __getitem__(self, index):
        return CuPyArray(self._data[index])

    def __rmul__(self, f: float):
        return CuPyArray(f * self._data)

    def __imul__(self, f: float):
        if isinstance(f, float):
            self._data *= f
        else:
            self._data *= f._data
        return self

    def __iadd__(self, other):
        self._data += other._data
        return self

    def __isub__(self, other):
        self._data -= other._data
        return self

    def __matmul__(self, other):
        return CuPyArray(self._data @ other._data)

    def ravel(self):
        return CuPyArray(self._data.ravel())

    def conj(self):
        return CuPyArray(self._data.conj())

    def reshape(self, shape):
        return CuPyArray(self._data.reshape(shape))

    def view(self, dtype):
        return CuPyArray(self._data.view(dtype))
