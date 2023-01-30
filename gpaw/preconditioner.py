import numpy as np

from gpaw.transformers import Transformer
from gpaw.fd_operators import Laplace

from gpaw.utilities.blas import axpy, scal
from gpaw.utilities.linalg import change_sign
from gpaw import extra_parameters
from gpaw import gpu
import _gpaw


class Preconditioner:
    def __init__(self, gd0, kin0, dtype=float, block=1, use_gpu=False):
        self.use_gpu = use_gpu
        gd1 = gd0.coarsen()
        gd2 = gd1.coarsen()
        self.kin0 = kin0
        self.kin1 = Laplace(gd1, -0.5, 1, dtype, use_gpu=self.use_gpu)
        self.kin2 = Laplace(gd2, -0.5, 1, dtype, use_gpu=self.use_gpu)
        self.scratch0 = gd0.zeros((2, block), dtype, False, use_gpu=self.use_gpu)
        self.scratch1 = gd1.zeros((3, block), dtype, False, use_gpu=self.use_gpu)
        self.scratch2 = gd2.zeros((3, block), dtype, False, use_gpu=self.use_gpu)
        self.step = 0.66666666 / kin0.get_diagonal_element()

        self.restrictor_object0 = Transformer(gd0, gd1, 1, dtype, use_gpu=self.use_gpu)
        self.restrictor_object1 = Transformer(gd1, gd2, 1, dtype, use_gpu=self.use_gpu)
        self.interpolator_object2 = Transformer(gd2, gd1, 1, dtype, use_gpu=self.use_gpu)
        self.interpolator_object1 = Transformer(gd1, gd0, 1, dtype, use_gpu=self.use_gpu)
        self.restrictor0 = self.restrictor_object0.apply
        self.restrictor1 = self.restrictor_object1.apply
        self.interpolator2 = self.interpolator_object2.apply
        self.interpolator1 = self.interpolator_object1.apply
        self.use_c_precond = extra_parameters.get('c_precond', True)

    def calculate_kinetic_energy(self, psit_xG, kpt):
        return None

    def __call__(self, residuals, kpt, ekin=None, out=None):
        if residuals.ndim == 3:
            if out is None:
                return self.__call__(residuals[None], kpt)[0]
            return self.__call__(residuals[None], kpt,
                                 out=out[None])[0]
        nb = len(residuals)  # number of bands
        phases = kpt.phase_cd
        step = self.step

        if self.use_gpu:
            # XXX GPUarray does not support properly multi-d slicing
            shape0 = (self.scratch0.shape[0],) + (nb,) + self.scratch0.shape[2:]
            shape1 = (self.scratch1.shape[0],) + (nb,) + self.scratch1.shape[2:]
            shape2 = (self.scratch2.shape[0],) + (nb,) + self.scratch2.shape[2:]

            if out is None:
                d0, q0 = gpu.array.get_slice(self.scratch0, shape0)
            else:
                d0 = out
                q0 = gpu.array.get_slice(self.scratch0, shape0)[0]
            r1, d1, q1 = gpu.array.get_slice(self.scratch1, shape1)
            r2, d2, q2 = gpu.array.get_slice(self.scratch2, shape2)

            self.restrictor0(residuals, r1, phases)
            change_sign(r1)
            gpu.memcpy_dtod(d1, r1, r1.nbytes)
            scal(4 * step, d1)
            self.kin1.apply(d1, q1, phases)
            q1 -= r1
            self.restrictor1(q1, r2, phases)
            gpu.memcpy_dtod(d2, r2, r2.nbytes)
            scal(16 * step, d2)
            self.kin2.apply(d2, q2, phases)
            q2 -= r2
            axpy(-16 * step, q2, d2)
            self.interpolator2(d2, q1, phases)
            d1 -= q1
            self.kin1.apply(d1, q1, phases)
            q1 -= r1
            axpy(-4 * step, q1, d1)
            self.interpolator1(d1, d0, phases)
            change_sign(d0)
            self.kin0.apply(d0, q0, phases)
            q0 -= residuals
            axpy(-step, q0, d0)  # d0 -= step * q0
            change_sign(d0)
            return d0

        if out is None:
            d0, q0 = self.scratch0[:, :nb]
        else:
            d0 = out
            q0 = self.scratch0[0, :nb]
        r1, d1, q1 = self.scratch1[:, :nb]
        r2, d2, q2 = self.scratch2[:, :nb]
        if self.use_c_precond:
            _gpaw.fd_precond(self.restrictor_object0.transformer,
                             self.restrictor_object1.transformer,
                             self.interpolator_object1.transformer,
                             self.interpolator_object2.transformer,
                             self.kin0.operator, self.kin1.operator,
                             self.kin2.operator,
                             d0, q0, r1, d1, q1, r2, d2, q2,
                             residuals, -residuals, step, phases)
            return d0
        self.restrictor0(-residuals, r1, phases)
        d1[:] = 4 * step * r1
        self.kin1.apply(d1, q1, phases)
        q1 -= r1
        self.restrictor1(q1, r2, phases)
        d2 = 16 * step * r2
        self.kin2.apply(d2, q2, phases)
        q2 -= r2
        d2 -= 16 * step * q2
        self.interpolator2(d2, q1, phases)
        d1 -= q1
        self.kin1.apply(d1, q1, phases)
        q1 -= r1
        d1 -= 4 * step * q1
        self.interpolator1(-d1, d0, phases)
        self.kin0.apply(d0, q0, phases)
        q0 -= residuals
        axpy(-step, q0, d0)  # d0 -= step * q0
        d0 *= -1.0
        return d0
