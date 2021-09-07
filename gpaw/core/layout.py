from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import DTypeLike
from gpaw.mpi import MPIComm, serial_comm
if TYPE_CHECKING:
    from gpaw.core.arrays import DistributedArrays


class Layout:
    def __init__(self, myshape):
        self.myshape = myshape
        self.dtype: DTypeLike

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
