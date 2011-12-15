# Written by Lauri Lehtovaara, 2008

"""This module defines BiCGStab-class, which implements biconjugate
gradient stabilized method. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.mpi import rank
from gpaw.tddft.linsolver import LinearSolver

class BiCGStab(LinearSolver):
    """Biconjugate gradient stabilized method
    
    This class solves a set of linear equations A.x = b using biconjugate
    gradient stabilized method (BiCGStab). The matrix A is a general,
    non-singular matrix, e.g., it can be nonsymmetric, complex, and
    indefinite. The method requires only access to matrix-vector product
    A.x = b, which is called A.dot(x,b). Thus A must provide the member
    function dot(self,x,b), where x and b are generally complex arrays
    numpy.array([...], dtype=complex), and x is the known vector, and
    b is the result.

    Now x and b are multivectors, i.e., list of vectors.
    """ 
    
    def __init__(self, sort_bands=True, tolerance=1e-8,
                 max_iterations=1000, eps=1e-15):
        """Create the BiCGStab-object.
        
        Tolerance should not be smaller than attainable accuracy, which is
        order of kappa(A) * eps, where kappa(A) is the (spectral) condition
        number of the matrix. The maximum number of iterations should be
        significantly less than matrix size, approximately
        .5 sqrt(kappa) ln(2/tolerance). A small number is treated as zero
        if it's magnitude is smaller than argument eps.
        
        Parameters
        ----------
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
        self.alpha_n = None
        self.omega_n = None
        self.p_nG = None
        self.v_nG = None
        self.q_nG = None
        self.r_nG = None
        self.work1_nG = None
        self.work2_nG = None

        LinearSolver.__init__(self, sort_bands, tolerance, max_iterations, eps)

        self.internals += ('scale_n', 'rhop_n', 'alpha_n', 'omega_n')

    def initialize(self, wfs):
        LinearSolver.initialize(self, wfs)

        nvec = self.bd.mynbands
        self.scale_n = np.empty(nvec, dtype=self.dtype)
        self.rhop_n = np.empty(nvec, dtype=self.dtype)
        self.alpha_n = np.empty(nvec, dtype=self.dtype)
        self.omega_n = np.empty(nvec, dtype=self.dtype)
        self.p_nG = self.gd.empty(nvec, dtype=self.dtype)
        self.v_nG = self.gd.empty(nvec, dtype=self.dtype)
        self.q_nG = self.gd.empty(nvec, dtype=self.dtype)
        self.r_nG = self.gd.empty(nvec, dtype=self.dtype)
        self.work1_nG = self.gd.empty(nvec, dtype=self.dtype)
        self.work2_nG = self.gd.empty(nvec, dtype=self.dtype)

        self.initialized = True

    def estimate_memory(self, mem):
        nvec = self.bd.mynbands
        gdbytes = self.gd.bytecount(self.dtype)

        mem.subnode('p_nG', nvec * gdbytes)
        mem.subnode('v_nG', nvec * gdbytes)
        mem.subnode('q_nG', nvec * gdbytes)
        mem.subnode('r_nG', nvec * gdbytes)
        mem.subnode('work1_nG', nvec * gdbytes)
        mem.subnode('work2_nG', nvec * gdbytes)

    def solve(self, A, x_nG, b_nG, slow_convergence_iters=50):
        """Solve a set of linear equations A.x = b.
        
        Parameters:
        A           matrix A
        x_nG        initial guess x_0 (on entry) and the result (on exit)
        b_nG        right-hand side (multi)vector

        """
        if not self.initialized:
            raise RuntimeError('BiCGStab: Solver has not been initialized.')

        self.timer.start('BiCGStab')

        # Multivector dot product, a^H b, where ^H is conjugate transpose
        def multi_zdotc(s, x,y, nvec):
            for i in range(nvec):
                s[i] = dotc(x[i],y[i])
            self.gd.comm.sum(s)
            return s
        # Multivector ZAXPY: a x + y => y
        def multi_zaxpy(a,x,y, nvec):
            for i in range(nvec):
                axpy(a[i]*(1+0J), x[i], y[i])
        # Multiscale: a x => x
        def multi_scale(a,x, nvec):
            for i in range(nvec):
                x[i] *= a[i]

        # Number of vectors to iterate on
        nvec = len(x_nG) #TODO ignore unoccupied bands
        assert nvec == self.bd.mynbands

        # Reset convergence flags and permutation
        self.conv_n[:] = False
        self.perm_n[:] = np.arange(nvec)

        # Use squared norm of b as scale
        multi_zdotc(self.scale_n, b_nG, b_nG, nvec)
        self.scale_n[:] = np.abs(self.scale_n)

        # Convergence check breaks down if scale < eps
        if np.any(self.scale_n < self.eps):
            raise RuntimeError('BiCGStab method detected underflow of squared '
                               'norm of right-hand side (scale = %le < '
                               'eps = %le).' % (self.scale_n.min(), eps))

        # r[0] = b - A x[0]
        A.dot(-x_nG, self.r_nG)
        self.r_nG += b_nG
        del b_nG

        # q = r[0]
        self.q_nG[:] = self.r_nG

        # rho[0] = alpha[0] = omega[0] = 1
        self.rhop_n[:] = 1.0
        self.alpha_n[:] = 1.0
        self.omega_n[:] = 1.0

        # p[0] = v[0] = 0
        self.p_nG[:] = 0.0
        self.v_nG[:] = 0.0

        for i in range(1, self.maxiter+1):
            # rho[i] = q^H r[i-1]
            rho_x = np.empty(nvec, dtype=self.dtype)
            multi_zdotc(rho_x, self.q_nG[:nvec], self.r_nG[:nvec], nvec)

            # beta = (rho[i] / rho[i-1]) (alpha[i-1] / omega[i-1])
            beta_x = (rho_x / self.rhop_n[:nvec]) * (self.alpha_n[:nvec] / self.omega_n[:nvec])

            # BiCGStab breaks down if abs(beta) / scale < eps
            if i > 1 and np.any(np.abs(beta_x) / self.scale_n[:nvec] < self.eps):
                raise RuntimeError('Biconjugate gradient stabilized method '
                                   'underflowed (|beta| = %le < eps = %le).'
                                    % (np.abs(beta_x).min(), self.eps))

            # p[i] = r[i-1] + beta * (p[i-1] - omega[i-1] * v[i-1])
            multi_zaxpy(-self.omega_n[:nvec], self.v_nG[:nvec], self.p_nG[:nvec], nvec)
            multi_scale(beta_x, self.p_nG[:nvec], nvec)
            self.p_nG[:nvec] += self.r_nG[:nvec]
            del beta_x

            # y = M^(-1) p[i]
            y_xG = self.work1_nG[:nvec]
            A.apply_preconditioner(self.p_nG[:nvec], y_xG)

            # v[i] = A y
            A.dot(y_xG, self.v_nG[:nvec])

            # alpha[i] = rho[i] / (q^H v[i])
            multi_zdotc(self.alpha_n[:nvec], self.q_nG[:nvec], self.v_nG[:nvec], nvec)
            self.alpha_n[:nvec] = rho_x / self.alpha_n[:nvec]

            # x[i] = x[i-1] + alpha[i] (M^(-1) p[i]) + omega[i] (M^(-1) s)
            # next line is the x[i] = x[i-1] + alpha[i] (M^(-1) p[i]) part
            multi_zaxpy(self.alpha_n[:nvec], y_xG, x_nG[:nvec], nvec)
            del y_xG

            # s = r[i-1] - alpha v[i]
            s_nG, self.r_nG = self.r_nG, None #XXX s now buffered in r
            multi_zaxpy(-self.alpha_n[:nvec], self.v_nG[:nvec], s_nG[:nvec], nvec)

            # Check convergence criteria |s|^2 < tol^2
            tmp_x = np.empty(nvec, dtype=self.dtype)
            multi_zdotc(tmp_x, s_nG[:nvec], s_nG[:nvec], nvec)
            self.conv_n[:nvec] = np.abs(tmp_x) / self.scale_n[:nvec] < self.tol**2
            if np.all(self.conv_n):
                self.r_nG, s_nG = s_nG, None #XXX s no longer buffered in r
                break

            # Print residuals if convergence is slow
            if i % slow_convergence_iters == 0:
                print 'Log10 S2 of proc #', rank, '  = ' , np.round(np.log10(np.abs(tmp_x)),1), \
                      ' after ', i, ' iterations'

            # z = M^(-1) s
            z_xG = self.work1_nG[:nvec]
            A.apply_preconditioner(s_nG[:nvec], z_xG)

            # t = A z
            t_xG = self.work2_nG[:nvec]
            A.dot(z_xG, t_xG)

            # omega[i] = t^H s / (t^H t)
            multi_zdotc(self.omega_n[:nvec], t_xG, s_nG[:nvec], nvec)
            multi_zdotc(tmp_x, t_xG, t_xG, nvec)
            self.omega_n[:nvec] /= tmp_x

            # x[i] = x[i-1] + alpha[i] (M^(-1) p[i]) + omega[i] (M^(-1) s)
            # next line is the x[i] = ... + omega[i] (M^-1 s) part
            multi_zaxpy(self.omega_n[:nvec], z_xG, x_nG[:nvec], nvec)
            del z_xG

            # r[i] = s - omega[i] * t
            self.r_nG, s_nG = s_nG, None #XXX s no longer buffered in r
            multi_zaxpy(-self.omega_n[:nvec], t_xG, self.r_nG[:nvec], nvec)
            del s_nG, t_xG

            # Check convergence criteria |r|^2 < tol^2
            multi_zdotc(tmp_x, self.r_nG[:nvec], self.r_nG[:nvec], nvec)
            self.conv_n[:nvec] = np.abs(tmp_x) / self.scale_n[:nvec] < self.tol**2
            if np.all(self.conv_n):
                break

            # Print residuals if convergence is slow
            if i % slow_convergence_iters == 0:
                print 'Log10 R2 of proc #', rank, '  = ' , np.round(np.log10(np.abs(tmp_x)),1), \
                      ' after ', i, ' iterations'

            # BiCGStab breaks down if abs(omega) / scale < eps
            if np.any(np.abs(self.omega_n[:nvec]) / self.scale_n[:nvec] < self.eps):
                raise RuntimeError('Biconjugate gradient stabilized method '
                                   'underflowed (|omega| = %le < eps = %le).'
                                    % (np.abs(self.omega_n).min(), self.eps))

            # Store rho for next iteration (rho[i-2] -> rho[i-1] etc.)
            self.rhop_n[:nvec] = rho_x
            del rho_x, tmp_x

            if self.sort_bands:
                # Move converged bands into one contiguous block at the end
                self.sort(self.p_nG, self.v_nG, self.q_nG, self.r_nG, x_nG)
                nvec = np.sum(~self.conv_n)

        self.niter = i

        if self.sort_bands:
            # Undo all permutations
            self.restore(x_nG)

        # Raise error if maximum number of iterations was reached
        if self.niter >= self.maxiter:
            raise RuntimeError('Biconjugate gradient stabilized method failed '
                               'to converge in %d iterations.' % self.maxiter)

        self.timer.stop('BiCGStab')

        return self.niter

