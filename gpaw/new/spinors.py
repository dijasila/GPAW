from __future__ import annotations

import numpy as np
from gpaw.core.domain import Domain
from gpaw.core.plane_waves import PWDesc
from gpaw.typing import Vector


class SpinorWaveFunctionDescriptor(Domain):
    def __init__(self,
                 pw: PWDesc,
                 qspiral_v: Vector | None = None):
        self.pw = pw
        self.qspiral_v = (np.asarray(qspiral_v) if qspiral_v is not None else
                          None)
        Domain.__init__(self, pw.cell_cv, pw.pbc_c, pw.kpt_c, pw.comm,
                        complex)
        self.myshape = (2,) + pw.myshape
        self.itemsize = pw.itemsize
        self.shape = (2,) + pw.shape
        self.dv = pw.dv

    def __repr__(self):
        q = self.qspiral_v
        return f'{self.__class__.__name__}({self.pw}, qspiral_v={q})'

    def new(self, *, kpt):
        pw = self.pw.new(kpt=kpt)
        pw.qspiral_v = self.qspiral_v
        return SpinorWaveFunctionDescriptor(pw, self.qspiral_v)

    def empty(self, shape, comm, xp=None):
        assert isinstance(shape, int)
        return self.pw.empty((shape, 2), comm)

    def global_shape(self) -> tuple[int, ...]:
        return (2,) + self.pw.global_shape()

    def indices(self, size):
        return self.pw.indices(size)
