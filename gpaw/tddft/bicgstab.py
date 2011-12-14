# Written by Lauri Lehtovaara, 2008

"""This module defines BiCGStab-class, which implements biconjugate
gradient stabilized method. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc
from gpaw.mpi import rank

class BiCGStab:
    """Biconjugate gradient stabilized method
    
    This class solves a set of linear equations A.x = b using biconjugate 
    gradient stabilized method (BiCGStab). The matrix A is a general, 
    non-singular matrix, e.g., it can be nonsymmetric, complex, and 
    indefinite. The method requires only access to matrix-vector product 
    A.x = b, which is called A.dot(x). Thus A must provide the member 
    function dot(self,x,b), where x and b are complex arrays 
    (numpy.array([], complex), and x is the known vector, and 
    b is the result.

    Now x and b are multivectors, i.e., list of vectors.
    """ 
    
    def __init__(self, gd, bd, allocate=False, timer=None,
                 tolerance=1e-15, max_iterations=1000, eps=1e-15):
        """Create the BiCGStab-object.
        
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

        self.allocated = False

        if eps <= tolerance:
            self.eps = eps
        else:
            raise RuntimeError('BiCGStab method got invalid tolerance (tol '
                               '= %le < eps = %le).' % (tolerance, eps))

        if allocate:
            self.allocate()

    def allocate(self):
        if self.allocated:
            return

        nvec = self.bd.mynbands

        self.scale_n = np.empty(nvec, dtype=complex)
        self.rhop_n = np.empty(nvec, dtype=complex)
        self.alpha_n = np.empty(nvec, dtype=complex)
        self.omega_n = np.empty(nvec, dtype=complex)
        self.p_nG = self.gd.empty(nvec, dtype=complex)
        self.v_nG = self.gd.empty(nvec, dtype=complex)
        self.q_nG = self.gd.empty(nvec, dtype=complex)
        self.r_nG = self.gd.empty(nvec, dtype=complex)
        self.work1_nG = self.gd.empty(nvec, dtype=complex)
        self.work2_nG = self.gd.empty(nvec, dtype=complex)

        self.allocated = True

    def estimate_memory(self, mem):
        nvec = self.bd.mynbands
        gdbytes = self.gd.bytecount(complex)

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
        if self.timer is not None:
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

        if not self.allocated:
            self.allocate()

        # Number of vectors to iterate on
        nvec = len(x_nG) #TODO ignore unoccupied bands
        assert nvec == self.bd.mynbands

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

        tmp_n = np.empty(nvec, dtype=complex)

        for i in range(1, self.maxiter+1):
            # rho[i] = q^H r[i-1]
            rho_n = np.empty(nvec, dtype=complex)
            multi_zdotc(rho_n, self.q_nG, self.r_nG, nvec)

            # beta = (rho[i] / rho[i-1]) (alpha[i-1] / omega[i-1])
            beta_n = (rho_n / self.rhop_n) * (self.alpha_n / self.omega_n)

            # BiCGStab breaks down if abs(beta) / scale < eps
            if i > 1 and np.any(np.abs(beta_n) / self.scale_n < self.eps):
                raise RuntimeError('Biconjugate gradient stabilized method '
                                   'underflowed (|beta| = %le < eps = %le).'
                                    % (np.abs(beta_n).min(), self.eps))

            # p[i] = r[i-1] + beta * (p[i-1] - omega[i-1] * v[i-1])
            multi_zaxpy(-self.omega_n, self.v_nG, self.p_nG, nvec)
            multi_scale(beta_n, self.p_nG, nvec)
            self.p_nG += self.r_nG
            del beta_n

            # y = M^(-1) p[i]
            y_nG = self.work1_nG
            A.apply_preconditioner(self.p_nG, y_nG)

            # v[i] = A y
            A.dot(y_nG, self.v_nG)

            # alpha[i] = rho[i] / (q^H v[i])
            multi_zdotc(self.alpha_n, self.q_nG, self.v_nG, nvec)
            self.alpha_n[:] = rho_n / self.alpha_n

            # x[i] = x[i-1] + alpha[i] (M^(-1) p[i]) + omega[i] (M^(-1) s)
            # next line is the x[i] = x[i-1] + alpha[i] (M^(-1) p[i]) part
            multi_zaxpy(self.alpha_n, y_nG, x_nG, nvec)
            del y_nG

            # s = r[i-1] - alpha v[i]
            s_nG, self.r_nG = self.r_nG, None #XXX s now buffered in r
            multi_zaxpy(-self.alpha_n, self.v_nG, s_nG, nvec)

            # Check convergence criteria |s|^2 < tol^2
            multi_zdotc(tmp_n, s_nG, s_nG, nvec)
            if np.all(np.abs(tmp_n) / self.scale_n < self.tol**2):
                self.r_nG, s_nG = s_nG, None #XXX s no longer buffered in r
                break

            # Print residuals if convergence is slow
            if i % slow_convergence_iters == 0:
                print 'Log10 S2 of proc #', rank, '  = ' , np.round(np.log10(np.abs(tmp_n)),1), \
                      ' after ', i, ' iterations'

            # z = M^(-1) s
            z_nG = self.work1_nG
            A.apply_preconditioner(s_nG, z_nG)

            # t = A z
            t_nG = self.work2_nG
            A.dot(z_nG, t_nG)

            # omega[i] = t^H s / (t^H t)
            multi_zdotc(self.omega_n, t_nG, s_nG, nvec)
            multi_zdotc(tmp_n, t_nG, t_nG, nvec)
            self.omega_n /= tmp_n

            # x[i] = x[i-1] + alpha[i] (M^(-1) p[i]) + omega[i] (M^(-1) s)
            # next line is the x[i] = ... + omega[i] (M^-1 s) part
            multi_zaxpy(self.omega_n, z_nG, x_nG, nvec)
            del z_nG

            # r[i] = s - omega[i] * t
            self.r_nG, s_nG = s_nG, None #XXX s no longer buffered in r
            multi_zaxpy(-self.omega_n, t_nG, self.r_nG, nvec)
            del s_nG, t_nG

            # Check convergence criteria |r|^2 < tol^2
            multi_zdotc(tmp_n, self.r_nG, self.r_nG, nvec)
            if np.all(np.abs(tmp_n) / self.scale_n < self.tol**2):
                break

            # Print residuals if convergence is slow
            if i % slow_convergence_iters == 0:
                print 'Log10 R2 of proc #', rank, '  = ' , np.round(np.log10(np.abs(tmp_n)),1), \
                      ' after ', i, ' iterations'

            # BiCGStab breaks down if abs(omega) / scale < eps
            if np.any(np.abs(self.omega_n) / self.scale_n < self.eps):
                raise RuntimeError('Biconjugate gradient stabilized method '
                                   'underflowed (|omega| = %le < eps = %le).'
                                    % (np.abs(self.omega_n).min(), self.eps))

            # Store rho for next iteration (rho[i-2] -> rho[i-1] etc.)
            self.rhop_n[:] = rho_n
            del rho_n

        self.niter = i

        # Raise error if maximum number of iterations was reached
        if self.niter >= self.maxiter:
            raise RuntimeError('Biconjugate gradient stabilized method failed '
                               'to converge in %d iterations.' % self.maxiter)

        if self.timer is not None:
            self.timer.stop('BiCGStab')

        return self.niter

