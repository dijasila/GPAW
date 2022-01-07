from __future__ import annotations

import numpy as np

from gpaw.core.arrays import DistributedArrays
from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import Array1D


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
        return (f'AtomDistribution(ranks={self.rank_a}, '
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

    def new(self, layout=None, data=None):
        return AtomArrays(layout or self.layout,
                          self.dims,
                          self.comm,
                          data=data,
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

    def gather(self, broadcast=False, copy=False) -> AtomArrays | None:
        assert not self.transposed
        comm = self.layout.atomdist.comm
        if comm.size == 1:
            if copy:
                aa = self.new()
                aa.data[:] = self.data
                return aa
            return self

        if comm.rank == 0 or broadcast:
            aa = self.new(layout=self.layout.new(atomdist=serial_comm))
        else:
            aa = None

        if comm.rank == 0:
            size_ra, size_r = self.sizes()
            shape = self.mydims + (size_r.max(),)
            buffer = np.empty(shape, self.layout.dtype)
            for rank in range(1, comm.size):
                buf = buffer[..., :size_r[rank]]
                comm.receive(buf, rank)
                b1 = 0
                for a, size in size_ra[rank].items():
                    b2 = b1 + size
                    aa[a] = buf[..., b1:b2].reshape(self.myshape +
                                                    self.layout.shape_a[a])
            for a, array in self._arrays.items():
                aa[a] = array
        else:
            comm.send(self.data, 0)

        if broadcast:
            comm.broadcast(aa.data, 0)

        return aa

    def sizes(self) -> tuple[list[dict[int, int]], Array1D]:
        comm = self.layout.atomdist.comm
        size_ra: list[dict[int, int]] = [{} for _ in range(comm.size)]
        size_r = np.zeros(comm.size, int)
        for a, (rank, shape) in enumerate(zip(self.layout.atomdist.rank_a,
                                              self.layout.shape_a)):
            size = np.prod(shape)
            size_ra[rank][a] = size
            size_r[rank] += size
        return size_ra, size_r

    def _dict_view(self):
        if self.transposed:
            return {a: np.moveaxis(array, 0, -1)
                    for a, array in self._arrays.items()}
        return self

    def scatter_from(self, data: np.ndarray = None) -> None:
        assert not self.transposed
        comm = self.layout.atomdist.comm
        if comm.size == 1:
            self.data[:] = data
            return

        if comm.rank != 0:
            comm.receive(self.data, 0, 42)
            return

        size_ra, size_r = self.sizes()
        aa = self.new(layout=self.layout.new(atomdist=serial_comm),
                      data=data)
        requests = []
        for rank, (totsize, size_a) in enumerate(zip(size_r, size_ra)):
            if rank != 0:
                buf = np.empty(self.mydims + (totsize,), self.layout.dtype)
                b1 = 0
                for a, size in size_a.items():
                    b2 = b1 + size
                    buf[..., b1:b2] = aa[a].reshape(self.mydims + (size,))
                request = comm.send(buf, rank, 42, False)
                # Remember to store a reference to the
                # send buffer (buf) so that is isn't
                # deallocated
                requests.append((request, buf))
            else:
                for a in size_a:
                    self[a] = aa[a]

        for request, _ in requests:
            comm.wait(request)
