from __future__ import annotations
import numpy as np
from gpaw.mpi import MPIComm, serial_comm
from gpaw.core.layout import Layout
from gpaw.matrix import Matrix


class DistributedArrays:
    def __init__(self,
                 layout: Layout,
                 shape: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None,
                 dtype = None,
                 layout_last: bool = True):
        self.layout = layout
        self.comm = comm
        self.layout_last = layout_last

        self.shape = shape if isinstance(shape, tuple) else (shape,)

        if self.shape:
            myshape0 = (self.shape[0] + comm.size - 1) // comm.size
            self.myshape = (myshape0,) + self.shape[1:]
        else:
            self.myshape = ()

        if layout_last:
            fullshape = self.myshape + layout.myshape
        else:
            fullshape = layout.myshape + self.myshape

        dtype = dtype or layout.dtype

        if data is not None:
            assert data.shape == fullshape
            assert data.dtype == dtype
        else:
            data = np.empty(fullshape, dtype)

        self.data = data

    def as_matrix(self):
        if self.layout_last:
            shape = (np.prod(self.shape), np.prod(self.layout.shape))
            myshape = (np.prod(self.myshape), np.prod(self.layout.myshape))
            dist = (self.comm, -1, 1)
        else:
            shape = (np.prod(self.layoutshape), np.prod(self.shape))
            myshape = (np.prod(self.layout.myshape), np.prod(self.myshape))
            dist = (self.comm, 1, -1)
        return Matrix(*shape,
                      data=self.data.reshape(myshape),
                      dist=dist)

    def __iadd__(self, other):
        other.acfs.add_to(self, other.coefs)
        return self
