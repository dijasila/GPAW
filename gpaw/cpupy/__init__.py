import numpy as np
import gpaw.cpupy.linalg as linalg
import gpaw.cpupy.cublas as cublas

__all__ = ['cublas']


class CuPyArray:
    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.size = data.size
        self.flags = data.flags

    @property
    def T(self):
        return CuPyArray(self.data.T)

    def __iter__(self):
        for data in self.data:
            yield CuPyArray(data)

    def __setitem__(self, index, value):
        if isinstance(value, CuPyArray):
            self.data[index] = value.data
        else:
            assert isinstance(value, float)
            self.data[index] = value

    def __getitem__(self, index):
        return CuPyArray(self.data[index])

    def __imul__(self, f: float):
        self.data *= f
        return self

    def __isub__(self, other):
        self.data -= other.data
        return self

    def __matmul__(self, other):
        return CuPyArray(self.data @ other.data)

    def ravel(self):
        return CuPyArray(self.data.ravel())

    def conj(self):
        return CuPyArray(self.data.conj())

    def reshape(self, shape):
        return CuPyArray(self.data.reshape(shape))

    def view(self, dtype):
        return CuPyArray(self.data.view(dtype))


def empty(*args, **kwargs):
    return CuPyArray(np.empty(*args, **kwargs))


def asnumpy(a, out=None):
    if out is None:
        return a.data.copy()
    out[:] = a.data
    return out


def asarray(a):
    return CuPyArray(a.copy())
