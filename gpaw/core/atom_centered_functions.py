from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.mpi import serial_comm, MPIComm
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.spline import Spline
from gpaw.typing import ArrayLike2D
if TYPE_CHECKING:
    from gpaw.core.uniform_grid import UniformGridFunctions


def to_spline(l, rcut, f):
    r = np.linspace(0, rcut, 100)
    return Spline(l, rcut, f(r))


class AtomCenteredFunctions:
    def __init__(self,
                 functions,
                 fracpos_ac: ArrayLike2D):
        self.functions = [[to_spline(*f) if isinstance(f, tuple) else f
                           for f in funcs]
                          for funcs in functions]
        self.fracpos_ac = np.array(fracpos_ac)

        self._layout = None
        self._lfc = None

    def __repr__(self):
        funcs = [['spdfgh'[f.l] for f in ff] for ff in self.functions[:4]]
        if len(self.functions) > 4:
            funcs.append(...)
        return f'{self.__class__.__name__}(functions={funcs})'

    @property
    def layout(self):
        self._lacy_init()
        return self._layout

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm,
              transposed=False) -> AtomArrays:
        return self.layout.empty(dims, comm, transposed=transposed)

    def move(self, fracpos_ac):
        self.fracpos_ac = np.array(fracpos_ac)
        self._lfc.set_positions(fracpos_ac)

    def add_to(self, functions, coefs=1.0):
        self._lacy_init()

        if isinstance(coefs, float):
            self._lfc.add(functions.data, coefs)
            return

        self._lfc.add(functions.data, coefs._dict_view(), q=0)

    def integrate(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.dims, functions.comm)
        self._lfc.integrate(functions.data, out._dict_view(), q=0)
        return out

    def derivative(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.dims + (3,), functions.comm,
                                    transposed=True)
        else:
            assert out.transposed
        coef_axiv = {a: np.moveaxis(array_ixv, 0, -2)
                     for a, array_ixv in out._arrays.items()}
        self._lfc.derivative(functions.data, coef_axiv, q=0)
        return out


class UniformGridAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, fracpos_ac, grid, integral=None, cut=False):
        AtomCenteredFunctions.__init__(self, functions, fracpos_ac)
        self.grid = grid
        self.integral = integral
        self.cut = cut

    def _lacy_init(self):
        if self._lfc is not None:
            return
        gd = self.grid._gd
        kd = KPointDescriptor(np.array([self.grid.kpt]))
        self._lfc = LFC(gd, self.functions, kd,
                        dtype=self.grid.dtype,
                        integral=self.integral,
                        forces=True,
                        cut=self.cut)
        self._lfc.set_positions(self.fracpos_ac)
        atomdist = AtomDistribution(
            ranks=np.array([sphere.rank for sphere in self._lfc.sphere_a]),
            comm=self.grid.comm)
        self._layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                         for funcs in self.functions],
                                        atomdist,
                                        self.grid.dtype)

    def to_uniform_grid(self,
                        out: UniformGridFunctions,
                        scale: float = 1.0) -> UniformGridFunctions:
        out.data[:] = 0.0
        self.add_to(out, scale)
        return out
