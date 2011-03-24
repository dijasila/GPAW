# -*- coding: utf-8 -*-
# Copyright (C) 2008  CSC Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.

"""This module defines an Overlap operator.

The module defines an overlap operator and implements overlap-related
functions.

"""
import sys
import numpy as np

from gpaw.utilities.tools import lowdin, tri2full
from gpaw import extra_parameters
from gpaw.utilities.lapack import diagonalize

class Overlap:
    """Overlap operator class.

    Attributes
    ==========

    dtype: type object
        Numerical type of operator (float/complex)

    """

    def __init__(self, ksl, timer):
        """Create the Overlap operator."""
        self.ksl = ksl
        self.timer = timer
        
    def orthonormalize(self, wfs, kpt):
        """Orthonormalizes the vectors a_nG with respect to the overlap.

        First, a Cholesky factorization C is done for the overlap
        matrix S_nn = <a_nG | S | a_nG> = C*_nn C_nn Cholesky matrix C
        is inverted and orthonormal vectors a_nG' are obtained as::

          psit_nG' = inv(C_nn) psit_nG
                    __
           ~   _   \    -1   ~   _
          psi (r) = )  C    psi (r)
             n     /__  nm     m
                    m

        Parameters
        ----------

        psit_nG: ndarray, input/output
            On input the set of vectors to orthonormalize,
            on output the overlap-orthonormalized vectors.
        kpt: KPoint object:
            k-point object from kpoint.py.
        work_nG: ndarray
            Optional work array for overlap matrix times psit_nG.
        work_nn: ndarray
            Optional work array for overlap matrix.

        """
        self.timer.start('Orthonormalize')
        psit_nG = kpt.psit_nG
        P_ani = kpt.P_ani
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)

        # Construct the overlap matrix:
        operator = wfs.matrixoperator

        def S(psit_G):
            return psit_G
        
        def dS(a, P_ni):
            return np.dot(P_ni, wfs.setups[a].dO_ii)

        self.timer.start('calc_matrix')

        S_nn = operator.calculate_matrix_elements(psit_nG, P_ani, S, dS)
        self.timer.stop('calc_matrix')

        orthonormalization_string = repr(self.ksl)
        self.timer.start(orthonormalization_string)
        #
        if extra_parameters.get('sic', False):
            #
            # symmetric Loewdin Orthonormalization
            tri2full(S_nn, UL='L', map=np.conj)
            nrm_n = np.empty(S_nn.shape[0])
            diagonalize(S_nn, nrm_n)
            nrm_nn = np.diag(1.0/np.sqrt(nrm_n))
            S_nn = np.dot(np.dot(S_nn.T.conj(), nrm_nn), S_nn)
        else:
            #
            self.ksl.inverse_cholesky(S_nn)
        # S_nn now contains the inverse of the Cholesky factorization.
        # Let's call it something different:
        C_nn = S_nn
        del S_nn
        self.timer.stop(orthonormalization_string)

        self.timer.start('rotate_psi')
        kpt.psit_nG = operator.matrix_multiply(C_nn, psit_nG, P_ani)
        self.timer.stop('rotate_psi')
        self.timer.stop('Orthonormalize')

    def apply(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply the overlap operator to a set of vectors.

        Parameters
        ==========
        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used

        """
        self.timer.start('Apply overlap')
        b_xG[:] = a_xG
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            P_axi[a] = np.dot(P_xi, wfs.setups[a].dO_ii)
            # gemm(1.0, wfs.setups[a].dO_ii, P_xi, 0.0, P_xi, 'n')
        wfs.pt.add(b_xG, P_axi, kpt.q) # b_xG += sum_ai pt^a_i P_axi
        self.timer.stop('Apply overlap')

    def apply_inverse(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply approximative inverse overlap operator to wave functions."""

        b_xG[:] = a_xG
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a,P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            P_axi[a] = np.dot(P_xi, wfs.setups[a].dC_ii)
        wfs.pt.add(b_xG, P_axi, kpt.q)

    def apply_inverse_cg(self, a_nG, x, wfs, kpt, calculate_P_ani = True):
        """ Apply the inverse overlap operator using the conjugate gradient method"""
        
        from gpaw.utilities.blas import dotu, axpy, dotc
        #from gpaw.tddft.cscg import multi_zdotu, multi_scale, multi_zaxpy
        #initialization
          # Multivector dot product, a^T b, where ^T is transpose
        def multi_zdotu(s, x,y, nvec):
            for i in range(nvec):
                s[i] = dotu(x[i],y[i])
            wfs.gd.comm.sum(s)
            return s
        # Multivector ZAXPY: a x + y => y
        def multi_zaxpy(a,x,y, nvec):
            for i in range(nvec):
                axpy(a[i]*(1+0J), x[i], y[i])
        # Multiscale: a x => x
        def multi_scale(a,x, nvec):
            for i in range(nvec):
                x[i] *= a[i]
        nvec = len(a_nG)
        r = wfs.gd.zeros(nvec, dtype=wfs.dtype)
        z  = wfs.gd.zeros((nvec,), dtype=wfs.dtype)
        sx = wfs.gd.zeros(nvec, dtype=wfs.dtype)
        p = wfs.gd.zeros(nvec, dtype=wfs.dtype)
        q = wfs.gd.zeros(nvec, dtype=wfs.dtype)
        alpha = np.zeros((nvec,), dtype=wfs.dtype) 
        beta = np.zeros((nvec,), dtype=wfs.dtype)
        scale = np.zeros((nvec,), dtype=wfs.dtype)
        normr2 = np.zeros((nvec,), dtype=wfs.dtype)
        rho  = np.zeros((nvec,), dtype=wfs.dtype) 
        rho_prev  = np.zeros((nvec,), dtype=wfs.dtype)
        rho_prev[:] = 1.0
        tol_cg = 1e-14
        multi_zdotu(scale, a_nG, a_nG, nvec)
        scale = np.abs(scale)
        #x0 = S^-1_approx a_nG
        self.apply_inverse(a_nG, x, wfs, kpt, calculate_P_ani)
        #r0 = a_nG - S x_0
        self.apply(-x, r, wfs, kpt, calculate_P_ani)
        r += a_nG
        #print 'r.max() =', abs(r).max()

        max_iter = 50

        for i in range(max_iter):

            #print 'iter =', i

            self.apply_inverse(r, z, wfs, kpt, calculate_P_ani)

            multi_zdotu(rho, r, z, nvec)

            beta = rho / rho_prev
            multi_scale(beta, p, nvec)
            p += z

            self.apply(p,q,wfs,kpt, calculate_P_ani)

            multi_zdotu(alpha, p, q, nvec)
            alpha = rho/alpha

            multi_zaxpy(alpha, p, x, nvec)
            multi_zaxpy(-alpha, q, r, nvec)

            multi_zdotu(normr2, r,r, nvec)

            #rhoc = rho.copy()
            rho_prev[:] = rho.copy()
            #rho.copy()
            #rho_prev = rho.copy()
            
            #print '||r|| =', np.sqrt(np.abs(normr2/scale))
            if ( (np.sqrt(np.abs(normr2) / scale)) < tol_cg ).all():
                break
        
            

