import numpy as npy
from math import pi
from FFT import fftnd
from gpaw.utilities.complex import real
from gpaw.poisson import PoissonSolver
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.tools import construct_reciprocal

class Coulomb:
    """Class used to evaluate coulomb integrals"""
    def __init__(self, gd, poisson=None):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gpaw module.
        """        
        self.gd = gd
        self.poisson = poisson

    def load(self, method):
        """Make sure all necessary attributes have been initialized"""
        
        # ensure that requested method is valid
        assert method in ('real', 'recip_gauss', 'recip_ewald'),\
            str(method) + ' is an invalid method name,\n' +\
            'use either real, recip_gauss, or recip_ewald'

        if method.startswith('recip'):
            if self.gd.comm.size > 1:
                raise RuntimeError('Cannot do parallel FFT, ' +\
                                   'use method=\'real\'')
            if not hasattr(self, 'k2'):
                self.k2, self.N3 = construct_reciprocal(self.gd)
                
            if method.endswith('ewald') and not hasattr(self, 'ewald'):
                # cutoff radius
                rc = 0.5 * npy.average(self.gd.domain.cell_c)
                # ewald potential: 1 - cos(k rc)
                self.ewald = npy.ones(self.gd.n_c) - \
                             npy.cos(npy.sqrt(self.k2)* rc)
                # lim k -> 0 ewald / k2 
                self.ewald[0, 0, 0] = .5 * rc**2

            if method.endswith('gauss') and not hasattr(self, 'ng'):
                gauss = Gaussian(self.gd)
                self.ng = gauss.get_gauss(0) / npy.sqrt(4 * pi)
                self.vg = gauss.get_gauss_pot(0) / npy.sqrt(4 * pi)
        
        else: # method == 'real'
            if not hasattr(self, 'solve'):
                if self.poisson is not None:
                    self.solve = self.poisson.solve
                else:
                    solver = PoissonSolver(nn=2)
                    solver.initialize(self.gd, load_gauss=True)
                    self.solve = solver.solve

    def coulomb(self, n1, n2=None, Z1=None, Z2=None, method='recip_gauss'):
        """Evaluates the coulomb integral of n1 and n2

        The coulomb integral is defined by::

                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------,
                      /    /         |r - r'|
                      
        where n1 and n2 could be complex.

        real:
           Evaluate directly in real space using gaussians to neutralize
           density n2, such that the potential can be generated by standard
           procedures
              
        recip_ewald:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by utilizing the Ewald /
           Tuckermann trick, which formaly requires the densities to be
           localized within half of the unit cell.

        recip_gauss:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by removing total charge
           of n1 and n2 with gaussian density ng::
           
                                                   *          *    *
            (n1|n2) = (n1 - Z1 ng|n2 - Z2 ng) + (Z2 n1 + Z1 n2 - Z1 Z2 ng | ng)

           The evaluation of the integral (n1 - Z1 ng|n2 - Z2 ng) is done in
           k-space using FFT techniques.
        """
        self.load(method)
        # determine integrand using specified method
        if method == 'real':
            I = self.gd.zeros()
            if n2 == None: n2 = n1; Z2 = Z1
            self.solve(I, n2, charge=Z2, eps=1e-12, zero_initial_phi=True)
            I *= npy.conjugate(n1)           
        elif method == 'recip_ewald':
            n1k = fftnd(n1)
            if n2 == None: n2k = n1k
            else: n2k = fftnd(n2)
            I = npy.conjugate(n1k) * n2k * \
                self.ewald * 4 * pi / (self.k2 * self.N3)
        elif method == 'recip_gauss':
            # Determine total charges
            if Z1 == None: Z1 = self.gd.integrate(n1)
            if Z2 == None and n2 != None: Z2 = self.gd.integrate(n2)

            # Determine the integrand of the neutral system
            # (n1 - Z1 ng)* int dr'  (n2 - Z2 ng) / |r - r'|
            nk1 = fftnd(n1 - Z1 * self.ng)
            if n2 == None:
                I = npy.absolute(nk1)**2 * 4 * pi / (self.k2 * self.N3)
            else:
                nk2 = fftnd(n2 - Z2 * self.ng)
                I = npy.conjugate(nk1) * nk2 * 4 * pi / (self.k2 * self.N3)

            # add the corrections to the integrand due to neutralization
            if n2 == None:
                I += (2 * real(npy.conjugate(Z1) * n1) - abs(Z1)**2 * self.ng)\
                     * self.vg
            else:
                I += (npy.conjugate(Z1) * n2 + Z2 * npy.conjugate(n1) -
                      npy.conjugate(Z1) * Z2 * self.ng) * self.vg
        else:
             raise RuntimeError, 'Method %s unknown' % method
         
        if n1.typecode() == npy.Float and (n2 == None or
                                           n2.typecode() == npy.Float):
            return real(self.gd.integrate(I))
        else:
            return self.gd.integrate(I)
