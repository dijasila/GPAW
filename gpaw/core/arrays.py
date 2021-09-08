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
                 data: np.ndarray = None):
        self.layout = layout
        self.comm = comm
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        if self.shape:
            myshape0 = (self.shape[0] + comm.size - 1) // comm.size
            self.myshape = (myshape0,) + self.shape[1:]
        else:
            self.myshape = ()

        fullshape = self.myshape + layout.myshape

        if data is not None:
            assert data.shape == fullshape
            assert data.dtype == layout.dtype
        else:
            data = np.empty(fullshape, layout.dtype)

        self.data = data

    def as_matrix(self):
        shape = (np.prod(self.shape), np.prod(self.layout.shape))
        myshape = (np.prod(self.myshape), np.prod(self.layout.myshape))
        return Matrix(*shape,
                      data=self.data.reshape(myshape),
                      dist=(self.comm, -1, 1))
