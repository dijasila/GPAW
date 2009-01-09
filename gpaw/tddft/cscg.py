# Written by Lauri Lehtovaara, 2008

"""This module defines CSCG-class, which implements conjugate gradient
for complex symmetric matrices. Requires Numpy and GPAW's own BLAS."""

import numpy as npy

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
    
    def __init__( self, gd, timer = None,
                  tolerance = 1e-15, max_iterations = 1000, eps=1e-15 ):
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
        
        self.gd = gd
        self.timer = timer
        

    def solve(self, A, x, b):
        """Solve a set of linear equations A.x = b.
        
        Parameters:
        A           matrix A
        x           initial guess x_0 (on entry) and the result (on exit)
        b           right-hand side (multi)vector

        """
        if self.timer is not None:
            self.timer.start('CSCG')

        # number of vectors
        nvec = len(x)

        # r(0) = b - A x(0)
        r = self.gd.zeros(nvec, dtype=complex)
        A.dot(-x,r)
        r += b

        # p(0) = r(0)
        p = self.gd.zeros(nvec, dtype=complex)
        p[:] = r

        # z(0) = M^(-1).r(0)
        z = self.gd.zeros(nvec, dtype=complex)
        A.apply_preconditioner(r,z)

        q = self.gd.zeros(nvec, dtype=complex)

        alpha = npy.zeros((nvec,), dtype=complex) 
        beta = npy.zeros((nvec,), dtype=complex) 
        rho  = npy.zeros((nvec,), dtype=complex) 
        rhop  = npy.zeros((nvec,), dtype=complex) 
        scale = npy.zeros((nvec,), dtype=complex) 
        tmp = npy.zeros((nvec,), dtype=complex) 

        # Multivector dot product, a^T b, where ^T is transpose
        def multi_zdotu(s, x,y, nvec):
            for i in range(nvec):
                s[i] = dotu(x[i],y[i])
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

        # rho(0) = r(0)^T z(0)
        multi_zdotu(rhop, r, z, nvec)

        # scale = square of the norm of b
        multi_zdotu(scale, b,b, nvec)
        scale = npy.abs( scale )

        # if scale < eps, then convergence check breaks down
        if (scale < self.eps).any():
            raise RuntimeError("CSCG method detected underflow for squared norm of right-hand side (scale = %le < eps = %le)." % (scale,eps))

        #print 'Scale = ', scale

        slow_convergence_iters = 1

        for i in range(self.max_iter):

            #print 'Beta = ', beta

            # if abs(beta) / scale < eps, then CSCG breaks down
            if ( (i > 0) and
                 ((npy.abs(beta) / scale) < self.eps).any() ):
                raise RuntimeError("Conjugate gradient method failed (abs(beta)=%le < eps = %le)." % (npy.min(npy.abs(beta)),self.eps))


            # q(i) = A.p(i)
            A.dot(p,q)

            # alpha(i) = rho(i) / (p(i)^T q(i))
            multi_zdotu(alpha, p, q, nvec)
            alpha = rhop / alpha

            # x(i+1) = x(i) + alpha(i) p(i)
            multi_zaxpy(alpha, p, x, nvec)

            # r(i+1) = r(i) - alpha(i) q(i)
            multi_zaxpy(-alpha, q, r, nvec)

            # z(i+1) = M^(-1).r(i+1)
            A.apply_preconditioner(r,z)

            # rho(i+1) = r(i+1)^T z(i+1)
            multi_zdotu(rho, r, z, nvec)

            # beta(i) = rho(i+1)/rho(i)
            beta = rho / rhop

            # p(i+1) = z(i+1) + beta(i) p(i)
            multi_scale(beta, p, nvec)
            p += z

            # if ( |r|^2 < tol^2 ) done
            multi_zdotu(tmp, r,r, nvec)
            if ( (npy.abs(tmp) / scale) < self.tol*self.tol ).all():
                #print 'R2 of proc #', rank, '  = ' , tmp, \
                #    ' after ', i+1, ' iterations'
                break

            # print if slow convergence
            if ((i+1) % slow_convergence_iters) == 0:
                print 'R2 of proc #', rank, '  = ' , '['+','.join(map(lambda v: '%.5g' % v,abs(tmp)))+']', \
                    ' after ', i+1, ' iterations'

            # finally update rho(i)->rho(i+1)
            rhop[:] = rho


        # if max iters reached, raise error
        if (i >= self.max_iter-1):
            raise RuntimeError("Conjugate gradient method failed to converged within given number of iterations (= %d)." % self.max_iter)


        # done
        self.iterations = i+1
        #print 'CSCG iterations = ', self.iterations

        if self.timer is not None:
            self.timer.stop('CSCG')

        return self.iterations
        #print self.iterations

