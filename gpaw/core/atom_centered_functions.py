from __future__ import annotations

import numpy as np
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import MPIComm, serial_comm
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.core.arrays import DistributedArrays
from gpaw.core.layout import Layout
from gpaw.lfc import LocalizedFunctionsCollection as LFC


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
        return self.f(r)


class AtomCenteredFunctions:
    def __init__(self, functions, positions, dtype, atomdist=serial_comm):
        self.functions = [[Function(*f) if isinstance(f, tuple) else f
                           for f in funcs]
                          for funcs in functions]
        self._positions = np.array(positions)

        self.layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                        for funcs in functions],
                                       atomdist,
                                       dtype)

        self.lfc = None

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value
        self.lfc.set_positions(value)

    def add_to(self, functions, coefs):
        self._lacy_init()
        self.lfc.add(functions.data, coefs, q=0)

    def integrate(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.shape, functions.comm)
        self.lfc.integrate(functions.data, out, q=0)
        return out


class UniformGridAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, positions, grid, atomdist=serial_comm):
        AtomCenteredFunctions.__init__(self, functions, positions, grid.dtype,
                                       atomdist)
        self.grid = grid

    def _lacy_init(self):
        if self.lfc is not None:
            return
        gd = self.grid._gd
        kd = KPointDescriptor(np.array([self.grid.kpt]))
        self.lfc = LFC(gd, self.functions, kd,
                       dtype=self.grid.dtype,
                       forces=True)
        self.lfc.set_positions(self._positions)


class PlaneWaveAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, positions, pw, atomdist=serial_comm):
        AtomCenteredFunctions.__init__(self, functions, positions, pw.dtype,
                                       atomdist)
        self.pw = pw

    def _lacy_init(self):
        if self.lfc is not None:
            return
        gd = self.pw.grid._gd
        kd = KPointDescriptor(np.array([self.pw.grid.kpt]))
        pd = PWDescriptor(self.pw.ecut, gd, kd=kd)
        self.lfc = PWLFC(self.functions, pd)
        self.lfc.set_positions(self._positions)


class AtomArraysLayout(Layout):
    def __init__(self,
                 shapes: list[int | tuple[int, ...]],
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

        Layout.__init__(self, (self.size,), (self.mysize,))

    def empty(self,
              shape: int | tuple[int, ...] = (),
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
                 shape: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, layout, shape, comm, data)
        self._arrays = {}
        for a, I1, I2 in layout.myindices:
            self._arrays[a] = self.data[..., I1:I2].reshape(
                self.myshape + layout.shapes[a])

    def __getitem__(self, a):
        return self._arrays[a]

    def get(self, a):
        return self._arrays.get(a)

    def __setitem__(self, a, value):
        self._arrays[a][:] = value

    def __contains__(self, a):
        return a in self._arrays
