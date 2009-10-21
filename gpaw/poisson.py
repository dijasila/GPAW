# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi
import sys

import numpy as np

from gpaw.transformers import Transformer
from gpaw.operators import Laplace, LaplaceA, LaplaceB
from gpaw import PoissonConvergenceError
from gpaw.utilities.blas import axpy
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.ewald import madelung
import gpaw.mpi as mpi
import _gpaw

class PoissonSolver:
    def __init__(self, nn='M', relax='GS', eps=2e-10):
        self.relax = relax
        self.nn = nn
        self.eps = eps
        self.charged_periodic_correction = None
        self.maxiter = 1000
        
        # Relaxation method
        if relax == 'GS':
            # Gauss-Seidel
            self.relax_method = 1
        elif relax == 'J':
            # Jacobi
            self.relax_method = 2
        else:
            raise NotImplementedError('Relaxation method %s' % relax)

    def set_grid_descriptor(self, gd):
        # Should probably be renamed initialize
        self.gd = gd
        self.dv = gd.dv

        gd = self.gd
        scale = -0.25 / pi 

        if self.nn == 'M':
            if gd.is_non_orthogonal():
                raise RuntimeError('Cannot use Mehrstellen stencil with '
                                   'non orthogonal cell.')

            self.operators = [LaplaceA(gd, -scale, allocate=False)]
            self.B = LaplaceB(gd, allocate=False)
        else:
            self.operators = [Laplace(gd, scale, self.nn, allocate=False)]
            self.B = None

        self.interpolators = []
        self.restrictors = []

        level = 0
        self.presmooths = [2]
        self.postsmooths = [1]
        
        # Weights for the relaxation,
        # only used if 'J' (Jacobi) is chosen as method
        self.weights = [2.0 / 3.0]
        
        while level < 4:
            try:
                gd2 = gd.coarsen()
            except ValueError:
                break
            self.operators.append(Laplace(gd2, scale, 1, allocate=False))
            self.interpolators.append(Transformer(gd2, gd, allocate=False))
            self.restrictors.append(Transformer(gd, gd2, allocate=False))
            self.presmooths.append(4)
            self.postsmooths.append(4)
            self.weights.append(1.0)
            level += 1
            gd = gd2

        self.levels = level

    def initialize(self, load_gauss=False):
        # Should probably be renamed allocate
        gd = self.gd
        self.rhos = [gd.empty()]
        self.phis = [None]
        self.residuals = [gd.empty()]
        for level in range(self.levels):
            gd2 = gd.coarsen()
            self.phis.append(gd2.empty())
            self.rhos.append(gd2.empty())
            self.residuals.append(gd2.empty())
            gd = gd2
        assert len(self.phis) == len(self.rhos)
        level += 1            
        assert level == self.levels
        
        for obj in self.operators + self.interpolators + self.restrictors:
            obj.allocate()
        if self.B is not None:
            self.B.allocate()
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()
        self.presmooths[level] = 8
        self.postsmooths[level] = 8

        if load_gauss:
            self.load_gauss()
        

    def load_gauss(self):
        if not hasattr(self, 'rho_gauss'):
            gauss = Gaussian(self.gd)
            self.rho_gauss = gauss.get_gauss(0)
            self.phi_gauss = gauss.get_gauss_pot(0)

    def solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
              zero_initial_phi=False):

        if eps is None:
            eps = self.eps

        actual_charge = self.gd.integrate(rho)
        background = (actual_charge / self.gd.dv /
                      self.gd.get_size_of_global_array().prod())

        if charge is None:
            charge = actual_charge
        if abs(charge) <= maxcharge:
            # System is charge neutral. Use standard solver
            return self.solve_neutral(phi, rho - background, eps=eps)
        
        elif abs(charge) > maxcharge and self.gd.pbc_c.all():
            # System is charged and periodic. Subtract a homogeneous
            # background charge
            if self.charged_periodic_correction is None:
                print "+-----------------------------------------------------+"
                print "| Calculating charged periodic correction using the   |"
                print "| Ewald potential from a lattice of probe charges in  |"
                print "| a homogenous background density                     |"
                print "+-----------------------------------------------------+"
                self.charged_periodic_correction = madelung(self.gd.cell_cv)
                print "Potential shift will be ", \
                      self.charged_periodic_correction , "Ha."
                       
            # Set initial guess for potential
            if zero_initial_phi:
                phi[:] = 0.0
            else:
                phi -= charge * self.charged_periodic_correction
            
            iters = self.solve_neutral(phi, rho - background, eps=eps)
            phi += charge * self.charged_periodic_correction
            return iters            
        
        elif abs(charge) > maxcharge and not self.gd.pbc_c.any():
            # The system is charged and in a non-periodic unit cell.
            # Determine the potential by 1) subtract a gaussian from the
            # density, 2) determine potential from the neutralized density
            # and 3) add the potential from the gaussian density.

            # Load necessary attributes
            self.load_gauss()

            # Remove monopole moment
            q = actual_charge / np.sqrt(4 * pi) # Monopole moment
            rho_neutral = rho - q * self.rho_gauss # neutralized density

            # Set initial guess for potential
            if zero_initial_phi:
                phi[:] = 0.0
            else:
                axpy(-q, self.phi_gauss, phi) #phi -= q * self.phi_gauss

            # Determine potential from neutral density using standard solver
            niter = self.solve_neutral(phi, rho_neutral, eps=eps)

            # correct error introduced by removing monopole
            axpy(q, self.phi_gauss, phi) #phi += q * self.phi_gauss
            
            return niter
        else:
            # System is charged with mixed boundaryconditions
            raise NotImplementedError
    
    def solve_neutral(self, phi, rho, eps=2e-10):
        self.phis[0] = phi

        if self.B is None:
            self.rhos[0][:] = rho
        else:
            self.B.apply(rho, self.rhos[0])
        
        niter = 1
        maxiter = self.maxiter
        while self.iterate2(self.step) > eps and niter < maxiter:
            niter += 1
        if niter == maxiter:
            charge = np.sum(rho.ravel()) * self.dv
            print 'CHARGE, eps:', charge, eps
            msg = 'Poisson solver did not converge in %d iterations!' % maxiter
            raise PoissonConvergenceError(msg)
        
        # Set the average potential to zero in periodic systems
        if np.alltrue(self.gd.pbc_c):
            phi_ave = self.gd.comm.sum(np.sum(phi.ravel()))
            N_c = self.gd.get_size_of_global_array()
            phi_ave /= np.product(N_c)
            phi -= phi_ave

        return niter
    
    def iterate(self, step, level=0):
        residual = self.residuals[level]
        niter = 0
        while True:
            niter += 1
            if level > 0 and niter == 1:
                residual[:] = -self.rhos[level]
            else:
                self.operators[level].apply(self.phis[level], residual)
                residual -= self.rhos[level]
            error = self.gd.comm.sum(np.vdot(residual, residual))
            if niter == 1 and level < self.levels:
                self.restrictors[level].apply(residual, self.rhos[level + 1])
                self.phis[level + 1][:] = 0.0
                self.iterate(4.0 * step, level + 1)
                self.interpolators[level].apply(self.phis[level + 1], residual)
                self.phis[level] -= residual
                continue
            residual *= step
            self.phis[level] -= residual
            if niter == 2:
                break
            
        return error
    
    def iterate2(self, step, level=0):
        """Smooths the solution in every multigrid level"""

        residual = self.residuals[level]

        if level < self.levels:
            self.operators[level].relax(self.relax_method,
                                        self.phis[level],
                                        self.rhos[level],
                                        self.presmooths[level],
                                        self.weights[level])

            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            self.restrictors[level].apply(residual,
                                          self.rhos[level + 1])
            self.phis[level + 1][:] = 0.0
            self.iterate2(4.0 * step, level + 1)
            self.interpolators[level].apply(self.phis[level + 1], residual)
            self.phis[level] -= residual

        self.operators[level].relax(self.relax_method,
                                    self.phis[level],
                                    self.rhos[level],
                                    self.postsmooths[level],
                                    self.weights[level])
        if level == 0:
            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            error = self.gd.comm.sum(np.dot(residual.ravel(),
                                             residual.ravel())) * self.dv
            return error

    def estimate_memory(self, mem):
        # XXX Memory estimate works only for J and GS, not FFT solver
        # Poisson solver appears to use same amount of memory regardless
        # of whether it's J or GS, which is a bit strange

        gdbytes = self.gd.bytecount()
        nbytes = -gdbytes # No phi on finest grid, compensate ahead
        for level in range(self.levels):
            nbytes += 3 * gdbytes # Arrays: rho, phi, residual
            gdbytes /= 8
        mem.subnode('rho, phi, residual [%d levels]' % self.levels, nbytes)

        for i, obj in enumerate(self.restrictors + self.interpolators):
            obj.estimate_memory(mem.subnode('Transformer %d' % i))

        for i, operator in enumerate(self.operators):
            name = operator.__class__.__name__
            operator.estimate_memory(mem.subnode('Operator %d [%s]' % (i, 
                                                                       name)))
        if self.B is not None:
            name = self.B.__class__.__name__
            self.B.estimate_memory(mem.subnode('B [%s]' % name))

    def __repr__(self):
        template = 'PoissonSolver(relax=\'%s\', nn=%s, eps=%e)'
        representation = template % (self.relax, repr(self.nn), self.eps)
        return representation


from numpy.fft import fftn, ifftn, fft2, ifft2
from gpaw.utilities.tools import construct_reciprocal


class PoissonFFTSolver(PoissonSolver):
    def __init__(self):
        self.charged_periodic_correction = None

    """FFT implementation of the poisson solver"""
    def initialize(self, gd, load_gauss=False):
        # XXX this won't work now, but supposedly this class will be deprecated
        # in favour of FFTPoissonSolver, no?
        self.gd = gd
        if self.gd.comm.size > 1:
            raise RuntimeError('Cannot do parallel FFT.')
        self.k2, self.N3 = construct_reciprocal(self.gd)
        if load_gauss:
            gauss = Gaussian(self.gd)
            self.rho_gauss = gauss.get_gauss(0)
            self.phi_gauss = gauss.get_gauss_pot(0)

    def solve_neutral(self, phi, rho, eps=None):
        phi[:] = np.real(ifftn(fftn(rho) * 4 * pi / self.k2))
        return 1

    def solve_screened(self, phi, rho, screening=0):
        phi[:] = np.real(ifftn(fftn(rho) * 4 * pi / (self.k2 + screening**2)))
        return 1


class FFTPoissonSolver(PoissonSolver):
    """FFT poisson-solver for general unit cells."""
    
    relax_method = 0
    nn = 999
    
    def __init__(self, eps=2e-10):
        self.charged_periodic_correction = None
        self.eps = eps

    def set_grid_descriptor(self, gd):
        assert gd.pbc_c.all()
        self.gd = gd

    def initialize(self):
        if self.gd.comm.rank == 0:
            self.k2_Q, self.N3 = construct_reciprocal(self.gd)

    def solve_neutral(self, phi_g, rho_g, eps=None):
        if self.gd.comm.size == 1:
            phi_g[:] = ifftn(fftn(rho_g) * 4.0 * pi / self.k2_Q).real
        else:
            rho_g = self.gd.collect(rho_g)
            if self.gd.comm.rank == 0:
                globalphi_g = ifftn(fftn(rho_g) * 4.0 * pi / self.k2_Q).real
            else:
                globalphi_g = None
            self.gd.distribute(globalphi_g, phi_g)
        return 1


class FixedBoundaryPoissonSolver(PoissonSolver):
    #solve the poisson equation with fft in two directions,
    #and with central differential method in the third direction.
    def __init__(self, nn=1):
        self.nn = nn
        assert self.nn == 1
        
    def set_grid_descriptor(self, gd):
        assert gd.pbc_c.all()
        self.gd = gd
        
    def initialize(self, b_phi1, b_phi2):
        if self.gd.comm.rank == 0: 
            d3 = b_phi1.shape[2]
            gd = self.gd
            N_c1 = gd.N_c[:2, np.newaxis]
            i_cq = np.indices(gd.N_c[:2]).reshape((2, -1))
            i_cq += N_c1 // 2
            i_cq %= N_c1
            i_cq -= N_c1 // 2
            B_vc = 2.0 * np.pi * gd.icell_cv.T[:2, :2]
            k_vq = np.dot(B_vc, i_cq) 
            k_vq *= k_vq
            self.k_vq2 = np.sum(k_vq, axis=0)
  
            b_phi1 = fft2(b_phi1, None, (0,1))
            b_phi2 = fft2(b_phi2, None, (0,1))
        
            self.b_phi1 = b_phi1[:, :, -1].reshape(-1)
            self.b_phi2 = b_phi2[:, :, 0].reshape(-1)
   
    def solve(self, phi_g, rho_g):
        actual_charge = self.gd.integrate(rho_g)
        background = (actual_charge / self.gd.dv /
                      self.gd.get_size_of_global_array().prod())
        return self.solve_neutral(phi_g, rho_g - background)
        
    def solve_neutral(self, phi_g, rho_g):
        # b_phi1 and b_phi2 are the boundary Hartree potential values
        # of left and right sides
        rho_g = self.gd.collect(rho_g)
      
        if self.gd.comm.rank == 0:
            d1, d2, d3 = rho_g.shape
            rho_g1 = fft2(rho_g, None, (0, 1))
            rho_g1 = rho_g1.reshape(d1 * d2, d3)        

            phi_g2 = self.gd.zeros(global_array=True, dtype=complex)
            phi_g2.shape = (d1 * d2, d3)
            du0 = np.zeros(d3 - 1, dtype=complex)
            du20 = np.zeros(d3 - 2, dtype=complex)       
        
            h2 = self.gd.h_c[2] ** 2
            for phi, rho, rv2, bp1, bp2, i in zip(phi_g2, rho_g1,
                                           self.k_vq2,
                                           self.b_phi1,
                                           self.b_phi2, range(d1*d2)):
                A = np.zeros(d3, dtype=complex) + 2 + h2 * rv2
                phi = rho * np.pi * 4 * h2
                phi[0] += bp1
                phi[-1] += bp2
                du = du0 - 1
                dl = du0 - 1
                du2 = du20 - 1            
                _gpaw.linear_solve_tridiag(d3, A, du, dl, du2, phi)
                phi_g2[i, :] = phi
     
            phi_g2.shape = (d1, d2, d3)
            globalphi_g = ifft2(phi_g2, None, (0, 1)).real
        else:
            globalphi_g = None
        self.gd.distribute(globalphi_g, phi_g)
        return 1
