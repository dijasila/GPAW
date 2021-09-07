from __future__ import annotations
from gpaw.mpi import MPIComm, serial_comm


class Layout:
    def __init__(self, myshape):
        self.myshape = myshape

    def zeros(self,
              shape: int | tuple[int] = (),
              comm: MPIComm = serial_comm):
        array = self.empty(shape, comm)
        array.data[:] = 0.0
        return array
