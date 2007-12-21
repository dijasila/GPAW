# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi

import numpy as npy

from gpaw.transformers import Transformer
from gpaw.operators import Laplace

from gpaw.utilities.blas import axpy

class Preconditioner:
    def __init__(self, gd0, kin0, typecode):
        gd1 = gd0.coarsen()
        gd2 = gd1.coarsen()
        self.kin0 = kin0
        self.kin1 = Laplace(gd1, -0.5, 1, typecode)
        self.kin2 = Laplace(gd2, -0.5, 1, typecode)
        self.scratch0 = gd0.zeros(2, typecode, False)
        self.scratch1 = gd1.zeros(3, typecode, False)
        self.scratch2 = gd2.zeros(3, typecode, False)
        self.step = 0.66666666 / kin0.get_diagonal_element()
        self.restrictor0 = Transformer(gd0, gd1, 1, typecode).apply
        self.restrictor1 = Transformer(gd1, gd2, 1, typecode).apply
        self.interpolator2 = Transformer(gd2, gd1, 1, typecode).apply
        self.interpolator1 = Transformer(gd1, gd0, 1, typecode).apply
        
    def __call__(self, residual, phases, phit, kpt):
        step = self.step
        d0, q0 = self.scratch0
        r1, d1, q1 = self.scratch1
        r2, d2, q2 = self.scratch2
        self.restrictor0(-residual, r1, phases)
        d1 = 4 * step * r1
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
        q0 -= residual
        axpy(-step, q0, d0)  # d0 -= step * q0
        d0 *= -1.0
        return d0


class Teter:
    def __init__(self, gd, kin, typecode):
        print 'Teter, Payne and Allan FFT Preconditioning of residue vector'
        self.typecode = typecode
        dims = gd.n_c.copy()
        dims.shape = (3, 1, 1, 1)
        icell = 1.0 / npy.array(gd.domain.cell_c)
        icell.shape = (3, 1, 1, 1)
        q_cq = ((npy.indices(gd.n_c) + dims / 2) % dims - dims / 2) * icell
        self.q2_q = npy.sum(q_cq**2)

        self.r_cG = npy.indices(gd.n_c, npy.Float) / dims
        self.r_cG.shape = (3, -1)

        self.cache = {}
        
    def __call__(self, R_G, phases, phit_G, kpt_c):
        from FFT import fftnd, inverse_fftnd
        if kpt_c is None:
            phit_q = fftnd(phit_G)
        else:
            phase_G = self.cache.get(kpt_c)
            if phase_G is None:
                phase_G = npy.exp(-2j * pi * npy.dot(kpt_c, self.r_cG))
                phase_G.shape = phit_G.shape
                self.cache[kpt_c] = phase_G
            phit_q = fftnd(phit_G * phase_G)
            
        norm = npy.vdot(phit_q, phit_q)
        h_q = phit_q * npy.conjugate(phit_q) * self.q2_q / norm
        ekin = npy.sum(h_q.ravel())
        x_q = self.q2_q / ekin
        
        K_q = x_q * 8.0
        K_q += 12.0
        K_q *= x_q
        K_q += 18.0
        K_q *= x_q
        K_q += 27.0
        K_q /= (K_q + 16.0 * x_q**4)
       
        if kpt_c is None:
            return inverse_fftnd(K_q * fftnd(R_G)).astype(npy.Float)
        else:
            return inverse_fftnd(K_q * fftnd(phase_G * R_G)) / phase_G
