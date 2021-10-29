from __future__ import annotations
import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.mpi import MPIComm, serial_comm


class AtomArraysLayout:
    def __init__(self,
                 shapes: list[int | tuple[int, ...]],
                 atomdist: AtomDistribution | MPIComm = serial_comm,
                 dtype=float):
        self.shapes = [shape if isinstance(shape, tuple) else (shape,)
                       for shape in shapes]
        if not isinstance(atomdist, AtomDistribution):
            atomdist = AtomDistribution(np.zeros(len(shapes), int), atomdist)
        self.atomdist = atomdist
        self.dtype = np.dtype(dtype)

        self.size = sum(np.prod(shape) for shape in self.shapes)

        self.myindices = []
        self.mysize = 0
        I1 = 0
        for a in atomdist.indices:
            I2 = I1 + np.prod(self.shapes[a])
            self.myindices.append((a, I1, I2))
            self.mysize += I2 - I1
            I1 = I2

        #Layout.__init__(self, (self.size,), (self.mysize,))

    def __repr__(self):
        return (f'AtomArraysLayout({self.shapes}, {self.atomdist}, '
                f'{self.dtype})')

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm,
              transposed=False) -> AtomArrays:
        return AtomArrays(self, dims, comm, transposed=transposed)


class AtomDistribution:
    def __init__(self, ranks, comm):
        self.comm = comm
        self.ranks = ranks
        self.indices = np.where(ranks == comm.rank)[0]

    def __repr__(self):
        return (f'AtomDistribution(ranks={self.ranks}, '
                f'comm={self.comm.rank}/{self.comm.size})')


class AtomArrays(DistributedArrays):
    def __init__(self,
                 layout: AtomArraysLayout,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None,
                 transposed=False):
        DistributedArrays. __init__(self, dims, (layout.mysize,),
                                    comm, layout.atomdist.comm,
                                    dtype=layout.dtype,
                                    data=data,
                                    dv=np.nan,
                                    transposed=transposed)
        self.layout = layout
        self._arrays = {}
        for a, I1, I2 in layout.myindices:
            self._arrays[a] = self.data[I1:I2].reshape(
                layout.shapes[a] + self.myshape)
        self.natoms: int = len(layout.shapes)

    def __repr__(self):
        return f'AtomArrays({self.layout})'

    def new(self):
        return AtomArrays(self.layout, self.dims, self.comm,
                          transposed=self.transposed)

    def __getitem__(self, a):
        return self._arrays[a]

    def get(self, a):
        return self._arrays.get(a)

    def __setitem__(self, a, value):
        self._arrays[a][:] = value

    def __contains__(self, a):
        return a in self._arrays

    def items(self):
        return self._arrays.items()

    def keys(self):
        return self._arrays.keys()

    def values(self):
        return self._arrays.values()

    def collect(self):
        assert self.layout.atomdist.comm.size == 1
        return self
