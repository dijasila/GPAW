from __future__ import annotations

import numpy as np
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import MPIComm, serial_comm
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.core.arrays import DistributedArrays
from gpaw.core.layout import Layout


class Function:
    def __init__(self, l, rcut, f):
        self.l = l
        self.rcut = rcut
        self.f = f

    def get_angular_momentum_number(self):
        return self.l

    def get_cutoff(self):
        return self.rcut

    def map(self, r):
        return self.d(r)


class PlaneWaveAtomCenteredFunctions:
    def __init__(self, functions, positions, pw, atomdist=serial_comm):
        self.functions = functions
        self.positions = np.array(positions)
        self.pw = pw

        self.layout = AtomArraysLayout([sum(2 * l + 1 for l, rc, f in funcs)
                                        for funcs in functions],
                                       atomdist,
                                       pw.grid.dtype)
        gd = pw.grid._gd
        kd = KPointDescriptor(np.array([pw.grid.kpt]))
        pd = PWDescriptor(pw.ecut, gd, kd=kd)
        self.lfc = PWLFC([[Function(*f) for f in funcs]
                          for funcs in functions],
                         pd)
        self.lfc.set_positions(self.positions)

    def add_to(self, functions, coefs):
        self.lfc.add(functions.data, coefs, q=0)


class AtomArraysLayout(Layout):
    def __init__(self,
                 shapes: list[int | tuple[int]],
                 atomdist: AtomDistribution | MPIComm = serial_comm,
                 dtype=float):
        self.shapes = [shape if isinstance(shape, tuple) else (shape,)
                       for shape in shapes]
        if not isinstance(atomdist, AtomDistribution):
            atomdist = AtomDistribution(np.zeros(len(shapes), int), atomdist)
        self.atomdist = atomdist
        self.dtype = dtype

        self.size = sum(np.prod(shape) for shape in self.shapes)

        self.myindices = []
        self.mysize = 0
        I1 = 0
        for a in atomdist.indices:
            I2 = I1 + np.prod(self.shapes[a])
            self.myindices.append((a, I1, I2))
            self.mysize += I2 - I1

        Layout.__init__(self, (self.mysize,))

    def empty(self,
              shape: int | tuple[int] = (),
              comm: MPIComm = serial_comm) -> AtomArrays:
        return AtomArrays(self, shape, comm)


class AtomDistribution:
    def __init__(self, ranks, comm):
        self.comm = comm
        self.ranks = ranks
        self.indices = np.where(ranks == comm.rank)[0]


class AtomArrays(DistributedArrays):
    def __init__(self,
                 layout: AtomArraysLayout,
                 shape: int | tuple[int] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, layout, shape, comm, data)
        self.layout = layout
        self._arrays = {}
        for a, I1, I2 in layout.myindices:
            self._arrays[a] = data[..., I1:I2].reshape(self.myshape +
                                                       layout.shapes[a])

    def __getitem__(self, a):
        return self._arrays[a]

    def __contains__(self, a):
        return a in self._arrays
