from __future__ import annotations
import numpy as np
from gpaw.mpi import MPIComm, serial_comm
from gpaw.core.arrays import DistributedArrays


class Layout:
    def __init__(self, myshape):
        self.myshape = myshape
        self.dtype: np.dtype

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> DistributedArrays:
        raise NotImplementedError

    def zeros(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> DistributedArrays:
        array = self.empty(shape, comm)
        array.data[:] = 0.0
        return array
