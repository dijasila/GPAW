# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements a class for (true) time-dependent density 
functional theory calculations."""

import sys

import Numeric as num

from gpaw.tddft.bicgstab import BiCGStab
from gpaw.tddft.propagators import \
    ExplicitCrankNicolson, \
    SemiImplicitCrankNicolson, \
    AbsorptionKick
from gpaw.tddft.tdopers import \
    TimeDependentHamiltonian, \
    TimeDependentOverlap, \
    TimeDependentDensity, \
    AbsorptionKickHamiltonian

class DummyMixer:
    def mix(self, nt_sG, comm):
        pass


class TDDFT:
    """ Time-dependent density functional theory
    
    This class is the core class of the time-dependent density functional 
    theory implementation and is the only class which user has to utilize.
    """
    
    def __init__( self, paw, td_potential = None, kpt = None,
                  propagator='ECN', solver='BiCGStab', tolerance=1e-10 ):
        """Create TDDFT-object.
        
        ============ =========================================================
        Parameters:
        ============ =========================================================
        paw          the PAW-object from a time-independent (the ground state)
                     calculation
        td_potential the time-dependent potential
        kpt          k-points   (if None, paw.kpt_u)
        propagator   the name of the time propagator
        solver       the name of the iterative linear equations solver 
        tolerance    tolerance for the solver
        ============ =========================================================
        
        Note: Use about ???10^-3 - 10^-4??? tighter tolerance for PAW.
        """
        
        # Convert PAW-object to complex
        paw.totype(num.Complex);

        # No density mixing
        paw.density.mixer = DummyMixer()

        # Set initial time
        self.time = 0.
        
        # Time-dependent variables and operators
        self.td_potential = td_potential
        self.td_hamiltonian = \
            TimeDependentHamiltonian( paw.pt_nuclei, 
                                      paw.hamiltonian, 
                                      td_potential )
        self.td_overlap = TimeDependentOverlap(paw.pt_nuclei)
        self.td_density = TimeDependentDensity(paw)
        
        # Grid descriptor
        self.gd = paw.gd

        # Solver for linear equations
        if solver is 'BiCGStab':
            self.solver = BiCGStab(gd=self.gd, tolerance=tolerance)
        else:
            raise RuntimeError( 'Error in TDDFT: Solver %s not supported. '
                                'Only BiCGStab is currently supported.' 
                                % (solver) )

        # Time propagator
        if propagator is 'ECN':
            self.propagator = \
                ExplicitCrankNicolson( self.td_density,
                                       self.td_hamiltonian,
                                       self.td_overlap,
                                       self.solver,
                                       self.gd )
        elif propagator is 'SICN':
            self.propagator = \
                SemiImplicitCrankNicolson( self.td_density, 
                                           self.td_hamiltonian, 
                                           self.td_overlap, 
                                           self.solver,
                                           self.gd )
        else:
            raise RuntimeError( 'Error in TDDFT:' + 
                                'Time propagator %s not supported. '
                                % (propagator) )
        
        # K-points        
        if kpt is not None:
            self.kpt = kpt
        else:
            self.kpt = paw.kpt_u

        # grid descriptor
        self.gd = paw.hamiltonian.gd
        # projectors
        self.pt_nuclei = paw.pt_nuclei

        
    def propagate(self, time_step, iterations=1):
        """Propagates wavefunctions.
        
        ============ =========================================================
        Parameters:
        ============ =========================================================
        time_step    time step
        iterations   iterations
        ============ =========================================================

        """
        for i in range(iterations):
            self.propagator.propagate(self.kpt, self.time, time_step)
            self.time += time_step
            

    # exp(ip.r) psi
    def absorption_kick(self, strength = 1e-3, direction = [0.0,0.0,1.0]):
        abs_kick = \
            AbsorptionKick( AbsorptionKickHamiltonian( self.pt_nuclei,
                                                       strength,
                                                       direction ),
                            self.td_overlap, self.solver, self.gd )
        abs_kick.kick(self.kpt)
