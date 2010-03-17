"""This module provides a class for linear density response calculations."""


import numpy as np

import ase.units as units

from gpaw.mixer import BaseMixer

from gpaw.dfpt.sternheimeroperator import SternheimerOperator
from gpaw.dfpt.linearsolver import LinearSolver
from gpaw.dfpt.scipylinearsolver import ScipyLinearSolver

__all__ = ["LinearResponse"]

class LinearResponse:
    """This class is a calculator for the sc density variation.

    From the given perturbation, the set of coupled equations for the
    first-order density response is calculated self-consistently.
    
    """
    
    def __init__(self, calc, perturbation):
        """Store calculator and init the LinearResponse calculator."""
        
        # Make sure that localized functions are initialized
        # calc.set_positions()
        self.calc = calc

        # Store grids
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd

        # wave function derivative
        self.psit1_unG = None

        self.sternheimer_operator = None
        self.linear_solver = None

        # 1) phonon
        # 2) constant E-field
        # 3) etc
        self.perturbation = perturbation

        self.initialized = False
        
    def initialize(self, tolerance_sternheimer = 1e-5):
        """Make the object ready for a calculation."""

        hamiltonian = self.calc.hamiltonian
        wfs = self.calc.wfs
        kpt_u = self.calc.wfs.kpt_u
        num_kpts = len(kpt_u)
        nbands = self.calc.wfs.nvalence/2

        # Initialize mixer
        self.mixer = BaseMixer(beta=0.4, nmaxold=5, weight=1)
        self.mixer.initialize(self.calc.density)
        self.mixer.reset()
        # Linear solver for the solution of Sternheimer equation
        self.linear_solver = ScipyLinearSolver(tolerance = tolerance_sternheimer)
        # Linear operator in the Sternheimer equation
        self.sternheimer_operator = SternheimerOperator(hamiltonian, wfs, self.gd)
        # List for storing the variations in the wave-functions
        self.psit1_unG = [np.array([self.gd.zeros() for n in range(nbands)])
                          for kpt in kpt_u]
        
        # Variation of the pseudo-potential
        self.vloc1_G = self.perturbation.calculate_derivative()
        self.vnl1_nG = self.perturbation.calculate_nonlocal_derivative(0)

        self.initialized = True
        
    def __call__(self, alpha = 0.4, max_iter = 1000, tolerance_sc = 1e-5,
                 tolerance_sternheimer = 1e-5):
        """Calculate linear density response.

        Implementation of q != 0 to be done!

        Parameters
        ----------
        alpha: float
            Linear mixing parameter
        max_iter: int
            Maximum number of iterations in the self-consistent evaluation of
            the density variation
        tolerance_sc: float
            Tolerance for the self-consistent loop measured in terms of
            integrated absolute change of the density derivative between two
            iterations
        tolerance_sternheimer: float
            Tolerance for the solution of the Sternheimer equation -- passed to
            the 'LinearSolver'
            
        """

        components = ['x','y','z']
        atoms = self.calc.get_atoms()
        symbols = atoms.get_chemical_symbols()
        print "Atom index: %i" % self.perturbation.a
        print "Atomic symbol: %s" % symbols[self.perturbation.a]
        print "Component: %s" % components[self.perturbation.v]
        
        self.initialize(tolerance_sternheimer)
        
        for iter in range(max_iter):
            print "iter:%3i" % iter,
            print "\tcalculating wave function variations"            
            if iter == 0:
                self.first_iteration()
            else:
                norm = self.iteration(iter, alpha)
                print "abs-norm: %6.3e\t" % norm,
                print "integrated density response: %5.2e" % \
                      self.gd.integrate(self.nt1_G)
        
                if norm < tolerance_sc:
                    print "self-consistent loop converged in %i iterations" \
                          % iter
                    break
            if iter == max_iter-1:
                print "self-consistent loop did not converge in %i iterations" \
                      % iter
                
        return self.nt1_G.copy(), self.psit1_unG
    
    def first_iteration(self):
        """Perform first iteration of sc-loop."""

        # Include only a fraction of the full local perturbation
        self.wave_function_variations(self.vloc1_G)
        self.nt1_G = self.density_response()
        self.mixer.mix(self.nt1_G, [])

    def iteration(self, iter, alpha):
        """Perform iteration.

        Parameters
        ----------
        iter: int
            Iteration number
        alpha: float
            Linear mixing parameter

        """

        # Copy old density
        # nt1_G_old = self.nt1_G.copy()
        # Update variation in the effective potential
        v1_G = self.effective_potential_variation()
        # Update wave function variations
        self.wave_function_variations(v1_G)
        # Update density
        self.nt1_G = self.density_response()
        # Mix
        self.mixer.mix(self.nt1_G, [])
        norm = self.mixer.get_charge_sloshing()
        # self.nt1_G = alpha * nt1_G + (1. - alpha) * nt1_G_old
        # Integrated absolute density change
        # norm = self.gd.integrate(np.abs(self.nt1_G - nt1_G_old))

        return norm

    def effective_potential_variation(self):
        """Calculate variation in the effective potential."""
        
        # Calculate new effective potential
        density = self.calc.density
        nt1_g = self.finegd.zeros()
        density.interpolator.apply(self.nt1_G, nt1_g)
        hamiltonian = self.calc.hamiltonian
        ps = hamiltonian.poisson
        # Hartree part
        vHXC1_g = self.finegd.zeros()
        ps.solve(vHXC1_g, nt1_g)
        # Store the Hartree potential from the density derivative
        # self.vH1_g = vHXC1_g.copy()
        # XC part
        nt_g_ = density.nt_g.ravel()
        vXC1_g = self.finegd.zeros()
        vXC1_g.shape = nt_g_.shape
        hamiltonian.xcfunc.calculate_fxc_spinpaired(nt_g_, vXC1_g)
        vXC1_g.shape = nt1_g.shape
        vHXC1_g += vXC1_g * nt1_g
        # Transfer to coarse grid
        v1_G = self.gd.zeros()
        hamiltonian.restrictor.apply(vHXC1_g, v1_G)
        # Add pseudo-potential part
        v1_G += self.vloc1_G

        return v1_G
    
    def wave_function_variations(self, v1_G):
        """Calculate variation in the wave-functions.

        Parameters
        ----------
        v1_G: ndarray
            Variation of the effective potential (PS + Hartree + XC)

        """

        nvalence = self.calc.wfs.nvalence
        kpt_u = self.calc.wfs.kpt_u
        # Calculate wave-function variations for all k-points.
        for kpt in kpt_u:

            k = kpt.k
            psit_nG = kpt.psit_nG[:nvalence/2]
            psit1_nG = self.psit1_unG[k]
            
            # Loop over all valence-bands
            for n in range(nvalence/2):

                # Get view of the Bloch function and its variation
                psit_G = psit_nG[n]
                psit1_G = psit1_nG[n]

                # Update k-point and band index in SternheimerOperator
                self.sternheimer_operator.set_blochstate(k, n)

                # rhs of Sternheimer equation
                rhs_G = -1. * v1_G * psit_G
                # Add non-local part
                rhs_G -= self.vnl1_nG[n]
                self.sternheimer_operator.project(rhs_G)
                
                print "\t\tBand %i -" % n,
                iter, info = self.linear_solver.solve(self.sternheimer_operator,
                                                      psit1_G, rhs_G)

                if info == 0:
                    print "linear solver converged in %i iterations" % iter
                elif info > 0:
                    print ("linear solver did not converge in %i iterations" %
                           iter)
                    assert info == 0
                else:
                    print "linear solver failed to converge" 
                    assert info == 0
                    
    def density_response(self):
        """Calculate density response from variation in the wave-functions."""

        nt1_G = self.gd.zeros()
    
        nbands = self.calc.wfs.nvalence/2
        kpt_u = self.calc.wfs.kpt_u

        for kpt in kpt_u:

            psit_nG = kpt.psit_nG[:nbands]
            psit1_nG = self.psit1_unG[kpt.k]

            for psit_G, psit1_G in zip(psit_nG, psit1_nG):

                nt1_G += 4 * psit_G * psit1_G

        return nt1_G
