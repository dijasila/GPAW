import numpy as np
import gpaw.gpu.cupy.linalg as linalg
import gpaw.gpu.cupy.cublas as cublas

__all__ = ['linalg', 'cublas']


def empty(*args, **kwargs):
    return ndarray(np.empty(*args, **kwargs))


def asnumpy(a, out=None):
    if out is None:
        return a._data.copy()
    out[:] = a._data
    return out


def asarray(a):
    return ndarray(a.copy())


def multiply(a, b, c):
    np.multiply(a._data, b._data, c._data)


def negative(a, b):
    np.negative(a._data, b._data)


def einsum(indices, *args, out=None):
    return ndarray(
        np.einsum(
            indices,
            *(arg._data for arg in args),
            out=None if out is None else out._data))


def diag(a):
    return ndarray(np.diag(a._data))


def eye(n):
    return ndarray(np.eye(n))


class ndarray:
    def __init__(self, data):
        assert isinstance(data, np.ndarray), type(data)
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.size = data.size
        self.flags = data.flags

    @property
    def T(self):
        return ndarray(self._data.T)

    @property
    def imag(self):
        return ndarray(self._data.imag)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for data in self._data:
            if data.ndim == 0:
                yield data.item()
            else:
                yield ndarray(data)

    def __setitem__(self, index, value):
        if isinstance(index, ndarray):
            index = index._data
        if isinstance(value, ndarray):
            self._data[index] = value._data
        else:
            assert isinstance(value, float)
            self._data[index] = value

    def __getitem__(self, index):
        return ndarray(self._data[index])

    def __mul__(self, f: float):
        if isinstance(f, float):
            return ndarray(f * self._data)
        return ndarray(f._data * self._data)

    def __rmul__(self, f: float):
        return ndarray(f * self._data)

    def __imul__(self, f: float):
        if isinstance(f, float):
            self._data *= f
        else:
            self._data *= f._data
        return self

    def __truediv__(self, other):
        return ndarray(self._data / other._data)

    def __add__(self, f):
        return ndarray(f._data + self._data)

    def __radd__(self, f):
        return ndarray(f + self._data)

    def __iadd__(self, other):
        self._data += other._data
        return self

    def __isub__(self, other):
        self._data -= other._data
        return self

    def __matmul__(self, other):
        return ndarray(self._data @ other._data)

    def ravel(self):
        return ndarray(self._data.ravel())

    def conj(self):
        return ndarray(self._data.conj())

    def reshape(self, shape):
        return ndarray(self._data.reshape(shape))

    def view(self, dtype):
        return ndarray(self._data.view(dtype))
