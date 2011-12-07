# Written by Lauri Lehtovaara, 2008

"""This module defines CSCG-class, which implements conjugate gradient
for complex symmetric matrices. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import dotc
from gpaw.mpi import rank

class CSCG:
    """Conjugate gradient for complex symmetric matrices
    
    This class solves a set of linear equations A.x = b using conjugate 
    gradient for complex symmetric matrices. The matrix A is a complex, 
    symmetric, and non-singular matrix. The method requires only access 
    to matrix-vector product A.x = b, which is called A.dot(x). Thus A 
    must provide the member function dot(self,x,b), where x and b are 
    complex arrays (numpy.array([], complex), and x is the known vector, 
    and b is the result.

    Now x and b are multivectors, i.e., list of vectors.
    """
    
    def __init__(self, gd, bd, allocate=False, timer=None,
                 tolerance=1e-15, max_iterations=1000, eps=1e-15):
        """Create the CSCG-object.
        
        Tolerance should not be smaller than attainable accuracy, which is 
        order of kappa(A) * eps, where kappa(A) is the (spectral) condition 
        number of the matrix. The maximum number of iterations should be 
        significantly less than matrix size, approximately 
        .5 sqrt(kappa) ln(2/tolerance). A small number is treated as zero
        if it's magnitude is smaller than argument eps.
        
        Parameters
        ----------
        gd: GridDescriptor
            grid descriptor for coarse (pseudowavefunction) grid
        bd: BandDescriptor
            band descriptor for state parallelization
        allocate: bool
            determines whether the constructor should allocate arrays
        timer: Timer
            timer
        tolerance: float
            tolerance for the norm of the residual ||b - A.x||^2
        max_iterations: integer
            maximum number of iterations
        eps: float
            if abs(rho) or omega < eps, it's regarded as zero 
            and the method breaks down

        """

        self.gd = gd
        self.bd = bd
        self.timer = timer        
        self.tol = tolerance
        self.maxiter = max_iterations
        self.niter = -1
        self.allocated = False

        if eps <= tolerance:
            self.eps = eps
        else:
            raise RuntimeError('CSCG method got invalid tolerance (tol = %le '
                               '< eps = %le).' % (tolerance, eps))

        if allocate:
            self.allocate()

    def allocate(self):
        if self.allocated:
            return

        nvec = self.bd.mynbands
        self.p_nG = self.gd.empty(nvec, dtype=complex)
        self.r_nG = self.gd.empty(nvec, dtype=complex)
        self.work_nG = self.gd.empty(nvec, dtype=complex)

        self.allocated = True

    def estimate_memory(self, mem):
        nvec = self.bd.mynbands
        gdbytes = self.gd.bytecount(complex)

        mem.subnode('p_nG', nvec * gdbytes)
        mem.subnode('r_nG', nvec * gdbytes)
        mem.subnode('work_nG', nvec * gdbytes)

    def solve(self, A, x_nG, b_nG):
        """Solve a set of linear equations A.x = b.
        
        Parameters:
        A           matrix A
        x_nG        initial guess x_0 (on entry) and the result (on exit)
        b_nG        right-hand side (multi)vector

        """
        if self.timer is not None:
            self.timer.start('CSCG')

        # Multivector dot product, a^T b, where ^T is transpose
        def multi_zdotu(s, x,y, nvec):
            for n in range(nvec):
                s[n] = dotu(x[n],y[n])
            self.gd.comm.sum(s)
            return s
        # Multivector ZAXPY: a x + y => y
        def multi_zaxpy(a,x,y, nvec):
            for n in range(nvec):
                axpy(a[n]*(1+0J), x[n], y[n])
        # Multiscale: a x => x
        def multi_scale(a,x, nvec):
            for n in range(nvec):
                x[n] *= a[n]

        if not self.allocated:
            self.allocate()

        # number of vectors
        nvec = len(x_nG)
        assert nvec == self.bd.mynbands

        slow_convergence_iters = 100

        # scale = square of the norm of b
        scale_n = np.empty((nvec,), dtype=complex)
        multi_zdotu(scale_n, b_nG, b_nG, nvec)
        scale_n = np.abs(scale_n)

        # if scale < eps, then convergence check breaks down
        if (scale_n < self.eps).any():
            raise RuntimeError('CSCG method detected underflow for squared '
                               'norm of right-hand side (scale = %le < '
                               'eps = %le).' % (scale_n, self.eps))

        # rho[-1] = 1 i.e. rhop[0] = 1
        rhop_n  = np.ones((nvec,), dtype=complex)

        # p[-1] = 0 i.e. p[0] = z
        self.p_nG.fill(0.0)

        # r[0] = b - A x[0]
        A.dot(-x_nG, self.r_nG)
        self.r_nG += b_nG

        for i in range(self.maxiter):
            # z = M^(-1) r[i]
            z_nG = self.work_nG
            A.apply_preconditioner(self.r_nG, z_nG)

            # rho[i] = r[i]^T z
            rho_n  = np.empty((nvec,), dtype=complex)
            multi_zdotu(rho_n, self.r_nG, z_nG, nvec)

            ##NB: beta=0 for i=0 if rho[0]=0 and rho[-1]=1 i.e. rhop[0]=1
            #if i == 0:
            #    # p[0] = z
            #    self.p_nG[:] = z_nG
            #else:
            #    # beta = rho[i] / rho[i-1]
            #    beta_n[:] = rho_n / rhop_n
            #    
            #    # p[i] = z + beta p[i-1]
            #    multi_scale(beta_n, self.p_nG, nvec)
            #    self.p_nG += z_nG

            # beta = rho[i] / rho[i-1]
            beta_n = rho_n / rhop_n

            # if abs(beta) / scale < eps, then CSCG breaks down
            if i > 0 and np.any(np.abs(beta_n) / scale_n < self.eps):
                raise RuntimeError('Conjugate gradient method failed '
                                   '(|beta| = %le < eps = %le).'
                                    % (np.min(np.abs(beta_n)), self.eps))

            # p[i] = z + beta p[i-1]
            multi_scale(beta_n, self.p_nG, nvec)
            self.p_nG += z_nG
            del z_nG, beta_n

            # q = A p[i]
            q_nG = self.work_nG
            A.dot(self.p_nG, q_nG)

            # alpha = rho[i] / (p[i]^T q)
            alpha_n = np.empty((nvec,), dtype=complex)
            multi_zdotu(alpha_n, self.p_nG, q_nG, nvec)
            alpha_n = rho_n / alpha_n

            # x[i] = x[i-1] + alpha p[i]
            multi_zaxpy(alpha_n, self.p_nG, x_nG, nvec)

            # r[i] = r[i-1] - alpha q
            multi_zaxpy(-alpha_n, q_nG, self.r_nG, nvec)
            del alpha_n, q_nG

            # if |r|^2 < tol^2: done
            r2_n = np.empty((nvec,), dtype=complex)
            multi_zdotu(r2_n, self.r_nG, self.r_nG, nvec)
            if np.all(np.abs(r2_n) / scale_n < self.tol**2):
                break

            # print if slow convergence
            if (i+1) % slow_convergence_iters == 0:
                print 'R2 of proc #', rank, '  = ' , r2_n, \
                    ' after ', i+1, ' iterations'

            # finally update rho (rho[i-2] -> rho[i-1] etc.)
            rhop_n[:] = rho_n
            del rho_n, r2_n

        if self.timer is not None:
            self.timer.stop('CSCG')

        return self.niter

