# Written by Lauri Lehtovaara, 2008

"""This module defines CSCG-class, which implements conjugate gradient
for complex symmetric matrices. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.utilities.linalg import change_sign
from gpaw.mpi import rank
from gpaw.tddft.utils import MultiBlas
from gpaw import gpu


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

    def __init__(self, gd, bd, timer=None,
                 tolerance=1e-15, max_iterations=1000, eps=1e-15,
                 blocksize=16, use_gpu=False):
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
            band descriptor
        timer: Timer
            timer
        tolerance: float
            tolerance for the norm of the residual ||b - A.x||^2
        max_iterations: integer
            maximum number of iterations
        eps: float
            if abs(rho) or (omega) < eps, it's regarded as zero
            and the method breaks down

        """

        self.tol = tolerance
        self.max_iter = max_iterations
        if eps <= tolerance:
            self.eps = eps
        else:
            raise RuntimeError(
                "CSCG method got invalid tolerance (tol = %le < eps = %le)." %
                (tolerance, eps))

        self.iterations = -1

        self.use_gpu = use_gpu

        self.gd = gd
        self.timer = timer
        self.mblas = MultiBlas(self.gd, timer)
        self.blocksize = min(blocksize, bd.mynbands)

        if self.use_gpu:
            cuda_blocks_min = 16
            cuda_blocks_max = 64
            self.blocksize = min(cuda_blocks_max, bd.mynbands,
                                 gd.comm.size * cuda_blocks_min,
                                 max(1, (224 * 224 * 224) * gd.comm.size
                                     / np.prod(gd.N_c)))

    def solve(self, A, x, b):
        """Solve a set of linear equations A.x = b.

        Parameters:
        A           matrix A
        x           initial guess x_0 (on entry) and the result (on exit)
        b           right-hand side (multi)vector

        """
        if self.timer is not None:
            self.timer.start('CSCG')

        on_gpu = gpu.is_device_array(x)

        # number of vectors
        nvec = len(x)

        B = min(self.blocksize, nvec)

        r = self.gd.empty(B, dtype=complex, use_gpu=on_gpu)
        p = self.gd.empty(B, dtype=complex, use_gpu=on_gpu)
        z = self.gd.empty(B, dtype=complex, use_gpu=on_gpu)

        if on_gpu:
            alpha = gpu.array.zeros((B,), dtype=complex)
            rho = gpu.array.zeros((B,), dtype=complex)
            rhop = gpu.array.zeros((B,), dtype=complex)
        else:
            alpha = np.zeros((B,), dtype=complex)
            rho = np.zeros((B,), dtype=complex)
            rhop = np.zeros((B,), dtype=complex)

        slow_convergence_iters = 100

        iterations = [];

        if self.timer is not None:
            self.timer.start('Iteration')

        for n1 in range(0, nvec, B):
            n2 = n1 + B
            if n2 > nvec:
                n2 = nvec
                B = n2 - n1
                r = r[:B]
                p = p[:B]
                z = z[:B]
                alpha = alpha[:B]
                rho = rho[:B]
                rhop = rhop[:B]

            x_x = x[n1:n2]
            b_x = b[n1:n2]

            rhop.fill(1.0)
            p.fill(0.0)

            # r_0 = b - A x_0
            A.dot(x_x, r)
            r -= b_x
            change_sign(r)

            # scale = square of the norm of b
            scale = self.mblas.multi_zdotc(b_x, b_x).real

            # if scale < eps, then convergence check breaks down
            if (scale < self.eps).any():
                raise RuntimeError("CSCG method detected underflow for squared norm of right-hand side (scale = %le < eps = %le)." % (np.min(scale), self.eps))

            for i in range(self.max_iter):
                # z_i = (M^-1.r)
                A.apply_preconditioner(r, z)

                # rho_i-1 = r^T z_i-1
                self.mblas.multi_zdotu(z, r, rho)

                # if i=1, p_i = r_i-1
                # else beta = (rho_i-1 / rho_i-2) (alpha_i-1 / omega_i-1)
                #      p_i = r_i-1 + b_i-1 (p_i-1 - omega_i-1 v_i-1)
                beta = rho / rhop

                #if abs(beta) / scale < eps, then CSCG breaks down
                if ( (i > 0) and
                     ((self.mblas.multi_zdotc(beta, beta).real / scale[n1:n2])
                      < self.eps).any() ):
                    raise RuntimeError("Conjugate gradient method failed (abs(beta)=%le < eps = %le)." % (np.min(self.mblas.multi_zdotc(beta, beta).real), self.eps))

                # p = z + beta p
                self.mblas.multi_scale(beta, p)
                p += z

                # z = A.p
                A.dot(p, z)

                # alpha_i = rho_i-1 / (p^T q_i)
                self.mblas.multi_zdotu(p, z, alpha)
                alpha = rho / alpha

                # x_i = x_i-1 + alpha_i p_i
                self.mblas.multi_zaxpy(alpha, p, x_x)
                # r_i = r_i-1 - alpha_i q_i
                self.mblas.multi_zaxpy(-alpha, z, r)

                # if ( |r|^2 < tol^2 ) done
                tmp = self.mblas.multi_zdotc(r, r).real
                if (tmp / scale < self.tol * self.tol).all():
                    #print 'R2 of proc #', rank, '  = ' , tmp, \
                    #    ' after ', i+1, ' iterations'
                    iterations.append(i)
                    break

                # finally update rho
                if gpu.is_device_array(rhop):
                    gpu.memcpy_dtod(rhop, rho, rho.nbytes)
                else:
                    rhop[:] = rho

                # print if slow convergence
                if ((i+1) % slow_convergence_iters) == 0:
                    print('R2 of proc #', rank, '  = ' , tmp, \
                          ' after ', i+1, ' iterations')

            # if max iters reached, raise error
            if (i >= self.max_iter - 1):
                raise RuntimeError("Conjugate gradient method failed to converged within given number of iterations (= %d)." % self.max_iter)

        self.iterations = np.max(iterations) + 1

        if self.timer is not None:
            self.timer.stop('Iteration')
            self.timer.stop('CSCG')

        return self.iterations
