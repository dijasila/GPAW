from __future__ import annotations

import numpy as np
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.core.matrix import Matrix
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.mpi import MPIComm, serial_comm
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.spline import Spline


def to_spline(l, rcut, f):
    r = np.linspace(0, rcut, 100)
    return Spline(l, rcut, f(r))


class AtomCenteredFunctions:
    def __init__(self,
                 functions,
                 positions,
                 dtype,
                 atomdist: AtomDistribution | MPIComm = serial_comm):
        self.functions = [[to_spline(*f) if isinstance(f, tuple) else f
                           for f in funcs]
                          for funcs in functions]
        self._positions = np.array(positions)

        self.layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                        for funcs in self.functions],
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

    def add_to(self, functions, coefs=1.0):
        self._lacy_init()

        if isinstance(coefs, float):
            self.lfc.add(functions.data, coefs)
            return

        self.lfc.add(functions.data,
                     {a: np.moveaxis(array, 0, -1)
                      for a, array in coefs._arrays.items()},
                     q=0)

    def integrate(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.shape, functions.comm)
        elif isinstance(out, Matrix):
            1 / 0
            out = AtomArrays(self.layout, functions.shape, functions.comm,
                             out.data)
        self.lfc.integrate(functions.data,
                           {a: np.moveaxis(array, 0, -1)
                            for a, array in out._arrays.items()},
                           q=0)
        return out

    def derivative(self, functions, out=None):
        self._lacy_init()
        if out is None:
            out = self.layout.empty(functions.shape + (3,), functions.comm)
        self.lfc.derivative(functions.data,
                            {a: np.moveaxis(array, 0, -2)
                             for a, array in out._arrays.items()},
                            q=-1)
        return out


class UniformGridAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, positions, grid, atomdist=serial_comm,
                 integral=None):
        AtomCenteredFunctions.__init__(self, functions, positions, grid.dtype,
                                       atomdist)
        self.grid = grid
        self.integral = integral

    def _lacy_init(self):
        if self.lfc is not None:
            return
        gd = self.grid._gd
        kd = KPointDescriptor(np.array([self.grid.kpt]))
        self.lfc = LFC(gd, self.functions, kd,
                       dtype=self.grid.dtype,
                       integral=self.integral,
                       forces=True)
        self.lfc.set_positions(self._positions)

    def evaluate(self, coef: float = 1.0):
        out = self.grid.zeros()
        self.add_to(out, coef)
        return out


class PlaneWaveAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, positions, pw, atomdist=serial_comm):
        AtomCenteredFunctions.__init__(self, functions, positions,
                                       pw.grid.dtype,
                                       atomdist)
        self.pw = pw

    def _lacy_init(self):
        if self.lfc is not None:
            return
        gd = self.pw.grid._gd
        kd = KPointDescriptor(np.array([self.pw.grid.kpt]))
        pd = PWDescriptor(self.pw.ecut, gd, kd=kd, dtype=self.pw.grid.dtype)
        self.lfc = PWLFC(self.functions, pd)
        self.lfc.set_positions(self._positions)

    def evaluate(self, coef: float = 1.0):
        out = self.pw.zeros()
        self.add_to(out, coef)
        return out.ifft()
