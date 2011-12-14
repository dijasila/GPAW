# Written by Lauri Lehtovaara, 2008

"""This module defines CSCG-class, which implements conjugate gradient
for complex symmetric matrices. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from gpaw import debug #XXX
from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import dotc
from gpaw.mpi import rank
from gpaw.tddft.linsolver import LinearSolver

class CSCG(LinearSolver):
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
    
    def __init__(self, gd, bd, timer, allocate=False, sort_bands=True,
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
        timer: Timer
            timer
        allocate: bool
            determines whether the constructor should allocate arrays
        sort_bands: bool
            determines whether to allow sorting of band by convergence
        tolerance: float
            tolerance for the norm of the residual ||b - A.x||^2
        max_iterations: integer
            maximum number of iterations
        eps: float
            if abs(rho) or omega < eps, it's regarded as zero
            and the method breaks down

        """

        self.scale_n = None
        self.rhop_n = None
        self.p_nG = None
        self.r_nG = None
        self.work_nG = None

        LinearSolver.__init__(self, gd, bd, timer, allocate, sort_bands,
                              tolerance, max_iterations, eps)

        self.internals += ('scale_n', 'rhop_n')

    def allocate(self):
        if self.allocated:
            return

        LinearSolver.allocate(self)

        nvec = self.bd.mynbands
        self.scale_n = np.empty(nvec, dtype=complex)
        self.rhop_n = np.empty(nvec, dtype=complex)
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

    def solve(self, A, x_nG, b_nG, slow_convergence_iters=100):
        """Solve a set of linear equations A.x = b.
        
        Parameters:
        A           matrix A
        x_nG        initial guess x_0 (on entry) and the result (on exit)
        b_nG        right-hand side (multi)vector

        """
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

        # Number of vectors to iterate on
        nvec = len(x_nG) #TODO ignore unoccupied bands
        assert nvec == self.bd.mynbands

        # Reset convergence flags and permutation
        self.conv_n[:] = False
        self.perm_n[:] = np.arange(nvec)

        # Use squared norm of b as scale
        multi_zdotu(self.scale_n, b_nG, b_nG, nvec)
        self.scale_n[:] = np.abs(self.scale_n)

        # Convergence check breaks down if scale < eps
        if np.any(self.scale_n < self.eps):
            raise RuntimeError('CSCG method detected underflow of squared '
                               'norm of right-hand side (scale = %le < '
                               'eps = %le).' % (self.scale_n.min(), self.eps))

        # rho[-1] = 1 i.e. rhop[0] = 1
        self.rhop_n[:] = 1.0

        # p[-1] = 0 i.e. p[0] = z
        self.p_nG[:] = 0.0

        # r[0] = b - A x[0]
        A.dot(-x_nG, self.r_nG)
        self.r_nG += b_nG
        del b_nG

        if debug:
            import time
            t = time.time()

        for i in range(1, self.maxiter+1):
            # z = M^(-1) r[i-1]
            z_xG = self.work_nG[:nvec]
            A.apply_preconditioner(self.r_nG[:nvec], z_xG)

            # rho[i] = r[i-1]^T z
            rho_x  = np.empty(nvec, dtype=complex)
            multi_zdotu(rho_x, self.r_nG[:nvec], z_xG, nvec)

            # beta = rho[i] / rho[i-1]
            beta_x = rho_x / self.rhop_n[:nvec]

            # CSCG breaks down if abs(beta) / scale < eps
            if i > 1 and np.any(np.abs(beta_x) / self.scale_n[:nvec] < self.eps):
                raise RuntimeError('Conjugate gradient method underflowed '
                                   '(|beta| = %le < eps = %le).'
                                    % (np.abs(beta_x).min(), self.eps))

            # p[i] = z + beta p[i-1]
            multi_scale(beta_x, self.p_nG[:nvec], nvec)
            self.p_nG[:nvec] += z_xG
            del z_xG, beta_x

            # q = A p[i]
            q_xG = self.work_nG[:nvec]
            A.dot(self.p_nG[:nvec], q_xG)

            # alpha = rho[i] / (p[i]^T q)
            alpha_x = np.empty(nvec, dtype=complex)
            multi_zdotu(alpha_x, self.p_nG[:nvec], q_xG, nvec)
            alpha_x = rho_x / alpha_x

            # x[i] = x[i-1] + alpha p[i]
            multi_zaxpy(alpha_x, self.p_nG[:nvec], x_nG[:nvec], nvec)

            # r[i] = r[i-1] - alpha q
            multi_zaxpy(-alpha_x, q_xG, self.r_nG[:nvec], nvec)
            del alpha_x, q_xG

            # Check convergence criteria |r|^2 < tol^2
            r2_x = np.empty(nvec, dtype=complex)
            multi_zdotu(r2_x, self.r_nG[:nvec], self.r_nG[:nvec], nvec)
            self.conv_n[:nvec] = np.abs(r2_x) / self.scale_n[:nvec] < self.tol**2
            if np.all(self.conv_n):
                break

            # Print residuals if convergence is slow
            if i % slow_convergence_iters == 0:
                print 'R2 of proc #', rank, '  = ' , r2_x, \
                    ' after ', i, ' iterations'

            if debug and self.gd.comm.rank == 0:
                print '----', '-' * self.bd.mynbands, 't=', time.time()-t
                print '%04d' % nvec, ''.join('1' if b else '0' for b in self.conv_n), self.perm_n
                t = time.time()

            # Store rho for next iteration (rho[i-2] -> rho[i-1] etc.)
            self.rhop_n[:nvec] = rho_x
            del rho_x, r2_x

            if self.sort_bands:
                # Move converged bands into one contiguous block at the end
                self.sort(self.p_nG, self.r_nG, x_nG)
                nvec = np.sum(~self.conv_n)

            if debug and self.gd.comm.rank == 0:
                print '%04d' % nvec, ''.join('1' if b else '0' for b in self.conv_n), self.perm_n

        self.niter = i

        if debug and self.gd.comm.rank == 0:
            print '----', '-' * self.bd.mynbands, 't=', time.time()-t

        if self.sort_bands:
            # Undo all permutations
            self.restore(x_nG)

        # Raise error if maximum number of iterations was reached
        if self.niter >= self.maxiter:
            raise RuntimeError('Conjugate gradient stabilized method failed '
                               'to converge in %d iterations.' % self.maxiter)

        self.timer.stop('CSCG')

        return self.niter

