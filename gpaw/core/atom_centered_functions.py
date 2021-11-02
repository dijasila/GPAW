from __future__ import annotations

import numpy as np
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.mpi import serial_comm, MPIComm
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.spline import Spline
from gpaw.typing import ArrayLike2D


def to_spline(l, rcut, f):
    r = np.linspace(0, rcut, 100)
    return Spline(l, rcut, f(r))


class AtomCenteredFunctions:
    def __init__(self,
                 functions,
                 fracpos: ArrayLike2D):
        self.functions = [[to_spline(*f) if isinstance(f, tuple) else f
                           for f in funcs]
                          for funcs in functions]
        self.fracpos = np.array(fracpos)

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

    def move(self, fracpos):
        self.fracpos = np.array(fracpos)
        self._lfc.set_positions(fracpos)

    def add_to(self, functions, coefs=1.0):
        self._lacy_init()

        if isinstance(coefs, float):
            self._lfc.add(functions.data, coefs)
            return

        if coefs.transposed:
            c = {a: np.moveaxis(array, 0, -1)
                 for a, array in coefs._arrays.items()}
        else:
            c = {a: array
                 for a, array in coefs._arrays.items()}
        self._lfc.add(functions.data, c, q=0)

    def integrate(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.dims, functions.comm)
        if out.transposed:
            c = {a: np.moveaxis(array, 0, -1)
                 for a, array in out._arrays.items()}
        else:
            c = {a: array
                 for a, array in out._arrays.items()}
        self._lfc.integrate(functions.data, c, q=0)
        return out

    def derivative(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.dims + (3,), functions.comm)
        self._lfc.derivative(functions.data,
                             {a: np.moveaxis(array, 0, -2)
                              for a, array in out._arrays.items()},
                             q=-1)
        return out


class UniformGridAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, fracpos, grid, integral=None, cut=False):
        AtomCenteredFunctions.__init__(self, functions, fracpos)
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
        self._lfc.set_positions(self.fracpos)
        atomdist = AtomDistribution(
            ranks=np.array([sphere.rank for sphere in self._lfc.sphere_a]),
            comm=self.grid.comm)
        self._layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                         for funcs in self.functions],
                                        atomdist,
                                        self.grid.dtype)

    def to_uniform_grid(self, coef: float = 1.0):
        out = self.grid.zeros()
        self.add_to(out, coef)
        return out
