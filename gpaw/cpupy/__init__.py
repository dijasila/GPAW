import numpy as np


class CuPyArray:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.size = data.size

    def __iter__(self):
        for data in self.data:
            yield CuPyArray(data)

    def __setitem__(self, index, value):
        assert isinstance(value, CuPyArray)
        assert index == slice(None)
        self.data[:] = value.data

    def __getitem__(self, index):
        return CuPyArray(self.data[index])

    def ravel(self):
        return CuPyArray(self.data.ravel())

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
