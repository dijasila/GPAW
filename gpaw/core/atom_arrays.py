from __future__ import annotations

from collections import defaultdict

import numpy as np

from gpaw.core.arrays import DistributedArrays
from gpaw.mpi import MPIComm, serial_comm


class AtomArraysLayout:
    def __init__(self,
                 shapes: list[int | tuple[int, ...]],
                 atomdist: AtomDistribution | MPIComm = serial_comm,
                 dtype=float):
        self.shape_a = [shape if isinstance(shape, tuple) else (shape,)
                        for shape in shapes]
        if not isinstance(atomdist, AtomDistribution):
            atomdist = AtomDistribution(np.zeros(len(shapes), int), atomdist)
        self.atomdist = atomdist
        self.dtype = np.dtype(dtype)

        self.size = sum(np.prod(shape) for shape in self.shape_a)

        self.myindices = []
        self.mysize = 0
        I1 = 0
        for a in atomdist.indices:
            I2 = I1 + np.prod(self.shape_a[a])
            self.myindices.append((a, I1, I2))
            self.mysize += I2 - I1
            I1 = I2

    def __repr__(self):
        return (f'AtomArraysLayout({self.shape_a}, {self.atomdist}, '
                f'{self.dtype})')

    def new(self, atomdist=None):
        return AtomArraysLayout(self.shape_a, atomdist or self.atomdist,
                                self.dtype)

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm,
              transposed=False) -> AtomArrays:
        return AtomArrays(self, dims, comm, transposed=transposed)


class AtomDistribution:
    def __init__(self, ranks, comm):
        self.comm = comm
        self.rank_a = ranks
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
            if transposed:
                self._arrays[a] = self.data[I1:I2].reshape(
                    layout.shape_a[a] + self.mydims)
            else:
                self._arrays[a] = self.data[..., I1:I2].reshape(
                    self.mydims + layout.shape_a[a])
        self.natoms: int = len(layout.shape_a)

    def __repr__(self):
        return f'AtomArrays({self.layout})'

    def new(self, layout=None):
        return AtomArrays(layout or self.layout, self.dims, self.comm,
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

    def collect(self, broadcast=False, copy=False):
        comm = self.layout.atomdist.comm
        if comm.size == 1:
            if copy:
                aa = self.new()
                aa.data[:] = self.data
                return aa
            return self

        if comm.rank == 0:
            size_ra = defaultdict(dict)
            size_r = defaultdict(int)
            for a, (rank, shape) in enumerate(zip(self.layout.atomdist.rank_a,
                                                  self.layout.shape_a)):
                size = np.prod(shape)
                size_ra[rank][a] = size
                size_r[rank] += size

            aa = self.new(layout=self.layout.new(atomdist=serial_comm))
            buffer = np.empty(max(size_r.values()), self.layout.dtype)
            for rank in range(1, comm.size):
                buf = buffer[:size_r[rank]]
                comm.receive(buf, rank)
                b1 = 0
                for a, size in size_ra[rank].items():
                    b2 = b1 + size
                    aa[a] = buf[b1:b2].reshape(self.layout.shape_a[a])
            for a, array in self._arrays.items():
                aa[a] = array
        else:
            comm.send(self.data.reshape((-1,)), 0)
            aa = None

        if broadcast:
            if comm.rank > 0:
                aa = self.new(layout=self.layout.new(atomdist=serial_comm))
            comm.broadcast(aa.data, 0)

        return aa

    def _dict_view(self):
        if self.transposed:
            return {a: np.moveaxis(array, 0, -1)
                    for a, array in self._arrays.items()}
        return self
