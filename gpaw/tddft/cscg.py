# Written by Lauri Lehtovaara, 2008

"""This module defines CSCG-class, which implements conjugate gradient
for complex symmetric matrices. Requires Numpy and GPAW's own BLAS."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotu
from gpaw.utilities.blas import dotc
from gpaw.utilities.linalg import change_sign
from gpaw.mpi import rank

from gpaw.tddft.utils import MultiBlas

import _gpaw

import gpaw.cuda


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
    
    def __init__( self, wfs, timer = None,
                  tolerance = 1e-15, max_iterations = 1000, eps=1e-15,
                  blocksize=16, cuda=False ):
        """Create the CSCG-object.
        
        Tolerance should not be smaller than attainable accuracy, which is 
        order of kappa(A) * eps, where kappa(A) is the (spectral) condition 
        number of the matrix. The maximum number of iterations should be 
        significantly less than matrix size, approximately 
        .5 sqrt(kappa) ln(2/tolerance). A small number is treated as zero
        if it's magnitude is smaller than argument eps.
        
        Parameters
        ----------
        wfs: Wavefunctions
            Coarse pseudowavefunctions
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
        if ( eps <= tolerance ):
            self.eps = eps
        else:
            raise RuntimeError("CSCG method got invalid tolerance (tol = %le < eps = %le)." % (tolerance,eps))

        self.iterations = -1

        self.cuda = cuda
        
        self.gd = wfs.gd
        self.timer = timer
        self.mblas = MultiBlas(self.gd, timer)
        
        self.blocksize = min(blocksize, wfs.bd.mynbands)
       
        if self.cuda:
            cuda_blocks_min = 16
            cuda_blocks_max = 64
            self.blocksize=min(cuda_blocks_max, wfs.bd.mynbands,
                               wfs.gd.comm.size*cuda_blocks_min,
                               max((224*224*224)*wfs.gd.comm.size/
                                   (wfs.gd.N_c[0]*wfs.gd.N_c[1]*wfs.gd.N_c[2]),1))
        

    def solve(self, A, x, b):
        """Solve a set of linear equations A.x = b.
        
        Parameters:
        A           matrix A
        x           initial guess x_0 (on entry) and the result (on exit)
        b           right-hand side (multi)vector

        """
        if self.timer is not None:
            self.timer.start('CSCG')

        cuda = isinstance(x,gpaw.cuda.gpuarray.GPUArray)

        # number of vectors
        nvec = len(x)

        B = min(self.blocksize, nvec)

        # r_0 = b - A x_0
        r = self.gd.empty(B, dtype=complex, cuda=cuda)
        p = self.gd.empty(B, dtype=complex, cuda=cuda)
        #q = self.gd.empty(B, dtype=complex, cuda=cuda)
        z = self.gd.empty(B, dtype=complex, cuda=cuda)

        if cuda:
            alpha = gpaw.gpuarray.zeros((B,), dtype=complex)
        #    beta = gpaw.gpuarray.zeros((B,), dtype=complex)
            rho  = gpaw.gpuarray.zeros((B,), dtype=complex)
            rhop  = gpaw.gpuarray.zeros((B,), dtype=complex)
        else:
            alpha = np.zeros((B,), dtype=complex)
        #    beta = np.zeros((B,), dtype=complex)
            rho  = np.zeros((B,), dtype=complex)
            rhop  = np.zeros((B,), dtype=complex)

        
        # r_0 = b - A x_0

        slow_convergence_iters = 100

        iterations = [];

        if self.timer is not None:
            self.timer.start('Iteration')

        for n1 in range(0, nvec, B):
            n2 = n1 + B
            if n2 > nvec:
                n2 = nvec
                B = n2 - n1
                #q = q[:B]
                r = r[:B]
                p = p[:B]
                z = z[:B]
                alpha = alpha[:B]
                #beta = beta[:B]
                rho = rho[:B]
                rhop = rhop[:B]

            x_x = x[n1:n2]
            b_x = b[n1:n2]

            # scale = square of the norm of b
            
            scale = self.mblas.multi_zdotc(b_x, b_x).real        
            # if scale < eps, then convergence check breaks down
            if (scale < self.eps).any():
                raise RuntimeError("CSCG method detected underflow for squared norm of right-hand side (scale = %le < eps = %le)." % (scale,eps))                    
            
            rhop.fill(1.0)
            p.fill(0.0)
            
            A.dot(x_x, r)
            r -= b_x
            change_sign(r)            
            for i in range(self.max_iter):                

                # z_i = (M^-1.r)
                A.apply_preconditioner(r,z)

                # rho_i-1 = r^T z_i-1
                self.mblas.multi_zdotu(z, r, rho)

                # if i=1, p_i = r_i-1
                # else beta = (rho_i-1 / rho_i-2) (alpha_i-1 / omega_i-1)
                #      p_i = r_i-1 + b_i-1 (p_i-1 - omega_i-1 v_i-1)
                beta = rho / rhop

                #if abs(beta) / scale < eps, then CSCG breaks down
                if ( (i > 0) and
                     (self.mblas.multi_zdotc(beta, beta).real /
                      scale[n1:n2]) < self.eps).any() ):
                    raise RuntimeError("Conjugate gradient method failed (abs(beta)=%le < eps = %le)." % (np.min(self.mblas.multi_zdotc(beta, beta).real)),self.eps))

                # p = z + beta p
                self.mblas.multi_scale(beta, p)
                p += z

                # q = A.p
                A.dot(p,z)

                #A.dot(p,q)

                # alpha_i = rho_i-1 / (p^T q_i)
                self.mblas.multi_zdotu(p, z, alpha)
                alpha = rho / alpha

                # x_i = x_i-1 + alpha_i p_i
                self.mblas.multi_zaxpy(alpha, p, x_x)
                # r_i = r_i-1 - alpha_i q_i
                self.mblas.multi_zaxpy(-alpha, z, r)
                #self.mblas.multi_zaxpy(-alpha, q, r)

                # if ( |r|^2 < tol^2 ) done
                    
                tmp = self.mblas.multi_zdotc(r, r).real
                
                if (tmp / scale <  self.tol*self.tol).all():
            
                    #if ( (np.abs(tmp) / scale[n1:n2]) < self.tol*self.tol ).all():
                    #print 'R2 of proc #', rank, '  = ' , tmp, \
                    #    ' after ', i+1, ' iterations'
                    iterations.append(i)
                    break

                # finally update rho
                if isinstance(rhop, gpaw.cuda.gpuarray.GPUArray):
                    gpaw.cuda.drv.memcpy_dtod(rhop.gpudata, rho.gpudata, rho.nbytes)
                else:
                    rhop[:] = rho

                # print if slow convergence
                if ((i+1) % slow_convergence_iters) == 0:
                    print 'R2 of proc #', rank, '  = ' , tmp, \
                          ' after ', i+1, ' iterations'



            # if max iters reached, raise error
            if (i >= self.max_iter-1):
                raise RuntimeError("Conjugate gradient method failed to converged within given number of iterations (= %d)." % self.max_iter)


        self.iterations = np.max(iterations) + 1

        #self.iterations = self.solve_blocks(A, x_gpu, b_gpu)

        if self.timer is not None:
            self.timer.stop('Iteration')
            
        if self.timer is not None:
            self.timer.stop('CSCG')

        return self.iterations

