# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi
import sys

import numpy as npy

from gpaw.transformers import Transformer
from gpaw.operators import Laplace, LaplaceA, LaplaceB
from gpaw import ConvergenceError
from gpaw.utilities.blas import axpy
from gpaw.utilities.gauss import Gaussian


class PoissonSolver:
    def __init__(self, nn='M', relax='GS'):
        self.nn = nn

        self.charged_periodic_correction = None
        
        # Relaxation method
        if relax == 'GS':
            # Gauss-Seidel
            self.relax_method = 1
        elif relax == 'J':
            # Jacobi
            self.relax_method = 2
        else:
            raise NotImplementedError('Relaxation method %s' % relax)

    def initialize(self, gd, load_gauss=False):
        self.gd = gd
        scale = -0.25 / pi 
        self.dv = gd.dv

        if self.nn == 'M':
            self.operators = [LaplaceA(gd, -scale)]
            self.B = LaplaceB(gd)
        else:
            self.operators = [Laplace(gd, scale, self.nn)]
            self.B = None

        self.rhos = [gd.empty()]
        self.phis = [None]
        self.residuals = [gd.empty()]
        self.interpolators = []
        self.restrictors = []

        level = 0
        self.presmooths = [2]
        self.postsmooths = [1]
        
        # Weights for the relaxation,
        # only used if 'J' (Jacobi) is chosen as method
        self.weights = [2.0/3.0]
        
        while level < 4:
            try:
                gd2 = gd.coarsen()
            except ValueError:
                break
            self.operators.append(Laplace(gd2, scale, 1))
            self.phis.append(gd2.empty())
            self.rhos.append(gd2.empty())
            self.residuals.append(gd2.empty())
            self.interpolators.append(Transformer(gd2, gd))
            self.restrictors.append(Transformer(gd, gd2))
            self.presmooths.append(4)
            self.postsmooths.append(4)
            self.weights.append(1.0)
            level += 1
            gd = gd2

        self.levels = level
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()
        self.presmooths[level] = 8
        self.postsmooths[level] = 8

        if load_gauss:
            gauss = Gaussian(self.gd)
            self.rho_gauss = gauss.get_gauss(0)
            self.phi_gauss = gauss.get_gauss_pot(0)

    def solve(self, phi, rho, eps=2e-10, charge=0, maxcharge=1e-6,
              zero_initial_phi=False):

        # Determine charge (if set to None)
        if charge == None:
            charge = self.gd.integrate(rho)

        if abs(charge) < maxcharge:
            # System is charge neutral. Use standard solver
            return self.solve_neutral(phi, rho, eps=eps)
        
        elif abs(charge) > maxcharge and npy.alltrue(
            self.gd.domain.pbc_c):
            # System is charged and periodic. Subtract a homogeneous
            # background charge
            background = charge / npy.product(self.gd.domain.cell_c)

            iters = self.solve_neutral(phi, rho - background, eps=eps)
            if self.charged_periodic_correction == None:
                # Load necessary attributes
                if not hasattr(self, 'rho_gauss'):
                    print "+---------------------------------------------------------------------------------------+"
                    print "| Calculating charged periodic correction according to eq 45 of ref:                    |"
                    print "| The Journal of Chemical Physics vol. 122, 243102, 2005                                |"
                    print "| ...using difference in energy between non periodic and periodic gaussian test charge. |"
                    print "+---------------------------------------------------------------------------------------+"
                    from gpaw.utilities.gauss import Gaussian
                    gauss = Gaussian(self.gd)
                    self.rho_gauss = gauss.get_gauss(0)
                    self.phi_gauss = gauss.get_gauss_pot(0)
                    periodic_gauss = self.gd.zeros()
                    neutral_gauss = self.rho_gauss / npy.sqrt(4*npy.pi) - 1. / npy.product(self.gd.domain.cell_c) 
                    self.solve_neutral(periodic_gauss, neutral_gauss, eps=1e-10)
                    E_periodic = self.gd.integrate(periodic_gauss * neutral_gauss)
                    E_single = self.gd.integrate(self.rho_gauss * self.phi_gauss) / (4*npy.pi)
                    self.charged_periodic_correction = (E_single - E_periodic)
                    print "Potential shift will be ", self.charged_periodic_correction, " Ha."
            
            iters = self.solve_neutral(phi, rho - background, eps=eps)
            phi += charge * self.charged_periodic_correction
            return iters            
        
        elif abs(charge) > maxcharge and not self.gd.domain.pbc_c.any():
            # The system is charged and in a non-periodic unit cell.
            # Determine the potential by 1) subtract a gaussian from the
            # density, 2) determine potential from the neutralized density
            # and 3) add the potential from the gaussian density.

            # Load necessary attributes
            if not hasattr(self, 'rho_gauss'):
                from gpaw.utilities.gauss import Gaussian
                gauss = Gaussian(self.gd)
                self.rho_gauss = gauss.get_gauss(0)
                self.phi_gauss = gauss.get_gauss_pot(0)

            # Remove monopole moment
            q = charge / npy.sqrt(4 * pi) # Monopole moment
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
        maxiter = 500
        while self.iterate2(self.step) > eps and niter < maxiter:
            niter += 1
        if niter == maxiter:
            charge = npy.sum(rho.ravel()) * self.dv
            print 'CHARGE, eps:', charge, eps
            raise ConvergenceError('Poisson solver did not converge!')

        # Set the average potential to zero in periodic systems
        if npy.alltrue(self.gd.domain.pbc_c):
            phi_ave = self.gd.comm.sum(npy.sum(phi.ravel()))
            N_c = self.gd.get_size_of_global_array()
            phi_ave /= npy.product(N_c)
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
            error = self.gd.comm.sum(npy.vdot(residual, residual))
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
            error = self.gd.comm.sum(npy.dot(residual.ravel(),
                                             residual.ravel())) * self.dv
            return error


from numpy.fft import fftn, ifftn
from gpaw.utilities.complex import real
from gpaw.utilities.tools import construct_reciprocal

class PoissonFFTSolver(PoissonSolver):
    def __init__(self):
        pass

    """FFT implementation of the poisson solver"""
    def initialize(self, gd, load_gauss=False):
        self.gd = gd
        if self.gd.domain.comm.size > 1:
            raise RuntimeError('Cannot do parallel FFT.')
        self.k2, self.N3 = construct_reciprocal(self.gd)
        if load_gauss:
            gauss = Gaussian(self.gd)
            self.rho_gauss = gauss.get_gauss(0)
            self.phi_gauss = gauss.get_gauss_pot(0)

    def solve_neutral(self, phi, rho, eps=None):
        phi[:] = real(ifftn(fftn(rho) * 4 * pi / self.k2))
        return 1

    def solve_screened(self, phi, rho, screening=0):
        phi[:] = real(ifftn(fftn(rho) * 4 * pi / (self.k2 + screening**2)))
        return 1
