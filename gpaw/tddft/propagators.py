# Written by Lauri Lehtovaara, 2007

"""This module implements time propagators for time-dependent density 
functional theory calculations."""

import sys

import numpy as npy

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc

from gpaw.mpi import rank, run

from gpaw.tddft.utils import MultiBlas


class DummyKPoint:
    def __init__(self):
        pass

###############################################################################
# DummyPropagator
###############################################################################
class DummyPropagator:
    """Time propagator
    
    The DummyPropagator-class is the VIRTUAL base class for all propagators.
    
    """
    def __init__(self, td_density, td_hamiltonian, td_overlap,
                solver, preconditioner, gd, timer, debug=True):
        """Create the DummyPropagator-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            the time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            the time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (/wavefunction) grid descriptor
        timer: Timer
            timer
        
        """ 
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap

        self.solver = solver
        self.preconditioner = preconditioner
        self.gd = gd
        self.timer = timer

        self.mblas = MultiBlas(gd)

        self.debug = debug #TODO! temporary

    #  Solve M psin = psi
    def apply_preconditioner(self, psi, psin):
        """Applies preconditioner.
        
        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result
        
        """
        self.timer.start('Solve TDDFT preconditioner')
        if self.preconditioner is not None:
            self.preconditioner.apply(self.kpt, psi, psin)
        else:
            psin[:] = psi
        self.timer.stop('Solve TDDFT preconditioner')

        
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions once. 
        
        Parameters
        ----------
        kpt_u: Kpoint
            K-point
        time: float
            the current time
        time_step: float
            the time step
        
        """ 
        raise RuntimeError( 'Error in DummyPropagator: '
                            'Member function propagate is virtual.' )


###############################################################################
# ExplicitCrankNicolson
###############################################################################
class ExplicitCrankNicolson(DummyPropagator):
    """Explicit Crank-Nicolson propagator
    
    Crank-Nicolson propagator, which approximates the time-dependent 
    Hamiltonian to be unchanged during one iteration step.
    
    (S(t) + .5 dt H(t) / hbar) psi(t+dt) = (S(t) - .5 dt H(t) / hbar) psi(t)
    
    """
    
    def __init__( self, td_density, td_hamiltonian, td_overlap, 
                  solver, preconditioner, gd, timer):
        """Create ExplicitCrankNicolson-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (/wavefunction) grid descriptor
        timer: Timer
            timer
        
        """
        DummyPropagator.__init__(self, td_density, td_hamiltonian, td_overlap,
                    solver, preconditioner, gd, timer)
        
        self.hpsit = None
        self.spsit = None
        

    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions. 
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points
        time: float
            the current time
        time_step: float
            time step
        
        """
        self.niter = 0
        self.timer.start('Update time-dependent operators')

        self.td_density.update()
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        self.td_overlap.update()

        if self.hpsit is None:
            self.hpsit = self.gd.zeros(n=len(kpt_u[0].psit_nG), dtype=complex)
        if self.spsit is None:
            self.spsit = self.gd.zeros(n=len(kpt_u[0].psit_nG), dtype=complex)

        self.timer.stop('Update time-dependent operators')

        # loop over k-points (spins)
        if self.debug: print '\nECN: Predictor step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        for kpt in kpt_u:
            self.solve_propagation_equation(kpt, time_step)

        return self.niter

    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def solve_propagation_equation(self, kpt, time_step):

        self.timer.start('Apply time-dependent operators')
        # Store H psi(t) as hpsit and S psit(t) as spsit
        run([nucleus.calculate_projections(kpt)
             for nucleus in self.td_hamiltonian.pt_nuclei])
        self.td_hamiltonian.apply(kpt, kpt.psit_nG, self.hpsit,
                                  calculate_P_uni=False)
        self.td_overlap.apply(kpt, kpt.psit_nG, self.spsit,
                              calculate_P_uni=False)
        self.timer.stop('Apply time-dependent operators')

        #psit[:] = self.spsit - .5J * self.hpsit * time_step
        kpt.psit_nG[:] = self.spsit
        self.mblas.multi_zaxpy(-.5j * time_step, self.hpsit, kpt.psit_nG, 
                                len(kpt.psit_nG))

        # Information needed by solver.solve -> self.dot
        self.kpt = kpt
        self.time_step = time_step

        # Solve A x = b where A is (S + i H dt/2) and b = kpt.psit_nG
        self.solver.solve(self, kpt.psit_nG, kpt.psit_nG)

    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunction.

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result ( S + i H dt/2 ) psi

        """
        self.timer.start('Apply time-dependent operators')
        run([nucleus.calculate_projections(self.kpt, psi)
             for nucleus in self.td_hamiltonian.pt_nuclei])
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit,
                                  calculate_P_uni=False)
        self.td_overlap.apply(self.kpt, psi, self.spsit, calculate_P_uni=False)
        self.timer.stop('Apply time-dependent operators')

        # psin[:] = self.spsit + .5J * self.time_step * self.hpsit 
        psin[:] = self.spsit
        self.mblas.multi_zaxpy(.5j * self.time_step, self.hpsit, psin, len(psi))


###############################################################################
# SemiImplicitCrankNicolson
###############################################################################
class SemiImplicitCrankNicolson(ExplicitCrankNicolson):
    """Semi-implicit Crank-Nicolson propagator
    
    Crank-Nicolson propagator, which first approximates the time-dependent 
    Hamiltonian to be unchanged during one iteration step to predict future 
    wavefunctions. Then the approximations for the future wavefunctions are
    used to approximate the Hamiltonian at the middle of the time step.
    
    (S(t) + .5 dt H(t) / hbar) psi_t(t+dt) = (S(t) - .5 dt H(t) / hbar) psi(t)
    (S(t) + .5 dt H(t+dt/2) / hbar) psi(t+dt) 
    = (S(t) - .5 dt H(t+dt/2) / hbar) psi(t)
    
    """
    
    def __init__( self, td_density, td_hamiltonian, td_overlap, 
                  solver, preconditioner, gd, timer):
        """Create SemiImplicitCrankNicolson-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            the time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            the time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (wavefunction) grid descriptor
        timer: Timer
            timer
        
        """
        ExplicitCrankNicolson.__init__(self, td_density, td_hamiltonian, 
                        td_overlap, solver, preconditioner, gd, timer)

        self.old_kpt_u = None
        self.tmp_kpt_u = None
        self.sinvhpsit = None


    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions once.
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points
        time: float
            the current time
        time_step: float
            time step
        """

        self.niter = 0

        # old/temporary wavefunctions
        if self.old_kpt_u is None:
            self.old_kpt_u = []
            for kpt in kpt_u:
                old_kpt = DummyKPoint()
                old_kpt.psit_nG = self.gd.empty( n=len(kpt.psit_nG),
                                                 dtype=complex )
                self.old_kpt_u.append(old_kpt)

        if self.tmp_kpt_u is None:
            self.tmp_kpt_u = []
            for kpt in kpt_u:
                tmp_kpt = DummyKPoint()
                tmp_kpt.psit_nG = self.gd.empty( n=len(kpt.psit_nG),
                                                 dtype=complex )
                self.tmp_kpt_u.append(tmp_kpt)

        if self.hpsit is None:
            self.hpsit = self.gd.zeros(len(kpt_u[0].psit_nG), dtype=complex)

        if self.spsit is None:
            self.spsit = self.gd.zeros(len(kpt_u[0].psit_nG), dtype=complex)

        #self.time_step = time_step #TODO! verify that commenting this out is dependency-safe


        self.timer.start('Update time-dependent operators')

        # Calculate electronic density rho(t) based on the wavefunctions psit_nG
        # in kpt_u valid for t = time. Updates nucleus.D_sp based on nucleus.P_uni.
        self.td_density.update()

        # Update Hamiltonian H(t) and overlap S(t) to reflect change
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        self.td_overlap.update() #TODO! Placement/effect?!

        self.timer.stop('Update time-dependent operators')

        # Copy current wavefunctions psit_nG to work and old wavefunction arrays
        for u in range(len(kpt_u)):
            self.tmp_kpt_u[u].psit_nG[:] = kpt_u[u].psit_nG
            self.old_kpt_u[u].psit_nG[:] = kpt_u[u].psit_nG

        # Predictor step
        # Here each psit_nG in tmp_kpt_u is overwritten by (1 - i S^(-1)(t) H(t) dt) psit_nG
        # from corresponding kpt_u as a full Euler step before psit(t+dt) is predicted.
        if self.debug: print '\nSICN: Predictor step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        for [kpt, rhs_kpt] in zip(kpt_u, self.tmp_kpt_u):
            self.solve_propagation_equation(kpt, rhs_kpt, time_step, guess=True)

        self.timer.start('Update time-dependent operators')

        # Calculate electronic density rho(t+dt) based on the wavefunctions psit_nG in
        # kpt_u valid for t = time+time_step. Updates nucleus.D_sp based on nucleus.P_uni.
        self.td_density.update()

        # Estimate Hamiltonian H(t+dt/2) and overlap S(t+dt/2) based on change
        self.td_hamiltonian.half_update( self.td_density.get_density(),
                                         time + time_step )
        self.td_overlap.half_update()

        self.timer.stop('Update time-dependent operators')

        # Corrector step
        # Each predicted psit_nG in kpt_u is given as an initial guess, whereas the old 
        # wavefunction in old_kpt_u are used to calculate rhs based on psit(t).
        if self.debug: print '\nSICN: Corrector step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        for [kpt, rhs_kpt] in zip(kpt_u, self.old_kpt_u):
            self.solve_propagation_equation(kpt, rhs_kpt, time_step)
        
        return self.niter


    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def solve_propagation_equation(self, kpt, rhs_kpt, time_step, guess=False):

        # kpt is guess, rhs_kpt is used to calculate rhs and is overwritten
        nvec = len(rhs_kpt.psit_nG)

        assert kpt != rhs_kpt, 'Data race condition detected'

        self.timer.start('Apply time-dependent operators')
        # Store H psi(t) as hpsit and S psit(t) as spsit
        run([nucleus.calculate_projections(kpt, rhs_kpt.psit_nG)
             for nucleus in self.td_hamiltonian.pt_nuclei])
        self.td_hamiltonian.apply(kpt, rhs_kpt.psit_nG, self.hpsit,
                                  calculate_P_uni=False)
        self.td_overlap.apply(kpt, rhs_kpt.psit_nG, self.spsit,
                              calculate_P_uni=False)
        self.timer.stop('Apply time-dependent operators')

        # Update rhs_kpt.psit_nG to reflect ( S - i H dt/2 ) psit(t)
        #rhs_kpt.psit_nG[:] = self.spsit - .5J * self.hpsit * time_step
        rhs_kpt.psit_nG[:] = self.spsit
        self.mblas.multi_zaxpy(-.5j * time_step, self.hpsit, rhs_kpt.psit_nG, nvec)

        if guess:
            if self.sinvhpsit is None:
                self.sinvhpsit = self.gd.zeros(len(kpt.psit_nG), dtype=complex)

            # Update kpt.psit_nG to estimate psit(t+dt) as ( 1 - i S^(-1) H dt ) psit(t)
            self.td_overlap.apply_inverse(kpt, self.hpsit, self.sinvhpsit)
            self.mblas.multi_zaxpy(-1.0j * time_step, self.sinvhpsit, kpt.psit_nG, nvec)

        # Information needed by solver.solve -> self.dot
        self.kpt = kpt
        self.time_step = time_step

        # Solve A x = b where A is (S + i H dt/2) and b = rhs_kpt.psit_nG
        self.niter += self.solver.solve(self, kpt.psit_nG, rhs_kpt.psit_nG)
                        
    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunctions.
        
        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result ( S + i H dt/2 ) psi
        
        """
        self.timer.start('Apply time-dependent operators')
        run([nucleus.calculate_projections(self.kpt, psi)
             for nucleus in self.td_hamiltonian.pt_nuclei])
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit,
                                  calculate_P_uni=False)
        self.td_overlap.apply(self.kpt, psi, self.spsit, calculate_P_uni=False)
        self.timer.stop('Apply time-dependent operators')

        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit
        psin[:] = self.spsit
        self.mblas.multi_zaxpy(.5j * self.time_step, self.hpsit, psin, len(psi))


###############################################################################
# DummyDensity
###############################################################################
class DummyDensity:
    """Implements dummy (= does nothing) density for AbsorptionKick."""
    def update(self):
        pass
        
    def get_density(self):
        return None

###############################################################################
# AbsorptionKick
###############################################################################
class AbsorptionKick(ExplicitCrankNicolson):
    """Absorption kick propagator
    
    Absorption kick propagator::

      (S(t) + .5 dt p.r / hbar) psi(0+) = (S(t) - .5 dt p.r / hbar) psi(0-)

    where ``|p| = (eps e / hbar)``, and eps is field strength, e is elementary 
    charge.
    
    """
    
    def __init__(self, abs_kick_hamiltonian, td_overlap, solver, preconditioner, gd, timer):
        """Create AbsorptionKick-object.
        
        Parameters
        ----------
        abs_kick_hamiltonian: AbsorptionKickHamiltonian
            the absorption kick hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (wavefunction) grid descriptor
        timer: Timer
            timer

        """
        ExpliticCrankNicolson.__init__(self, DummyDensity(), abs_kick_hamiltonian,
                        td_overlap, solver, preconditioner, gd, timer)


    def kick(self, kpt_u):
        """Excite all possible frequencies.
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points

        """ 

        # if rank == 0:
        #     self.text('Kick iterations = ', self.td_hamiltonian.iterations)

        for l in range(self.td_hamiltonian.iterations):
            self.propagate(kpt_u, 0, 1.0)
            # if rank == 0:
            #     self.text('.')
        # if rank == 0:
        #     print ''
        

###############################################################################
# SemiImpicitTaylorExponential
###############################################################################
class SemiImplicitTaylorExponential(DummyPropagator):
    """Semi-implicit Taylor exponential propagator 
    exp(-i S^-1 H t) = 1 - i S^-1 H t + (1/2) (-i S^-1 H t)^2 + ...  
    
    """
    
    def __init__( self, td_density, td_hamiltonian, td_overlap, solver, 
                  preconditioner, degree, gd, timer ):
        """Create SemiImplicitTaylorExponential-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            the time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            the time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner
        degree: integer
            Degree of the Taylor polynomial
        gd: GridDescriptor
            coarse (wavefunction) grid descriptor
        timer: Timer
            timer
        
        """
        DummyPropagator.__init__(self, td_density, td_hamiltonian, td_overlap,
                                solver, preconditioner, gd, timer)

        self.degree = degree

        self.tmp_kpt_u = None
        self.psin = None
        self.hpsit = None
        
        
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions once.
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points
        time: float
            the current time
        time_step: float
            time step
        """

        self.niter = 0
        # temporary wavefunctions
        if self.tmp_kpt_u is None:
            self.tmp_kpt_u = []
            for kpt in kpt_u:
                tmp_kpt = DummyKPoint()
                tmp_kpt.psit_nG = self.gd.empty( n=len(kpt.psit_nG),
                                                 dtype=complex )
                self.tmp_kpt_u.append(tmp_kpt)


        # Allocate memory 
        nvec = len(kpt_u[0].psit_nG)
        if self.psin is None:
            self.psin = self.gd.zeros( nvec, 
                                       dtype=complex )
        if self.hpsit is None:
            self.hpsit = self.gd.zeros( nvec, 
                                       dtype=complex )
        
        #self.time_step = time_step #TODO! verify that commenting this out is dependency-safe


        self.timer.start('Update time-dependent operators')

        # rho(t)
        self.td_density.update()
        # H(t)
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        # S(t)
        self.td_overlap.update()

        self.timer.stop('Update time-dependent operators')


        # copy current wavefunctions to temporary variable
        for u in range(len(kpt_u)):
            self.tmp_kpt_u[u].psit_nG[:] = kpt_u[u].psit_nG

        # predict for each kpt_u
        if self.debug: print '\nSITE: Predictor step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        for kpt in kpt_u:
            self.solve_propagation_equation(kpt, time_step)


        self.timer.start('Update time-dependent operators')

        # rho(t+dt)
        self.td_density.update()
        # H(t+dt/2)
        self.td_hamiltonian.half_update( self.td_density.get_density(),
                                         time + time_step )
        # S(t+dt/2)
        self.td_overlap.half_update()

        self.timer.stop('Update time-dependent operators')


        # propagate psit(t), not psit(t+dt), in correct
        for u in range(len(kpt_u)):
            kpt_u[u].psit_nG[:] = self.tmp_kpt_u[u].psit_nG

        # correct for each kpt_u
        if self.debug: print '\nSITE: Corrector step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        for kpt in kpt_u:
            self.solve_propagation_equation(kpt, time_step)
        
        return self.niter

    # psi(t) = exp(-i t S^-1 H) psi(0)
    # psi(t) = 1  + (-i S^-1 H t) (1 + (1/2) (-i S^-1 H t) (1 + ... ) )
    def solve_propagation_equation(self, kpt, time_step):

        nvec = len(kpt.psit_nG)

        # Information needed by solver.solve -> self.dot
        self.kpt = kpt
        self.time_step = time_step
            
        # psin = psi(0)
        self.psin[:] = kpt.psit_nG

        for k in range(self.degree,0,-1):
            # psin = psi(0) + (1/k) (-i S^-1 H t) psin
            self.td_hamiltonian.apply(kpt, self.psin, self.hpsit)
            # S psin = H psin
            self.psin[:] = self.hpsit
            self.niter += self.solver.solve(self, self.psin, self.hpsit)
            #print 'Linear solver iterations = ', self.solver.iterations
            # psin = psi(0) + (-it/k) S^-1 H psin
            self.mblas.multi_scale( -(1.0J) * time_step / k, 
                                     self.psin, nvec )
            self.mblas.multi_zaxpy(1.0, kpt.psit_nG, self.psin, nvec)

        kpt.psit_nG[:] = self.psin

    def dot(self, psit, spsit):
        self.td_overlap.apply(self.kpt, psit, spsit)


###############################################################################
# SemiImpicitKrylovExponential
###############################################################################
class SemiImplicitKrylovExponential(DummyPropagator):
    """Semi-implicit Krylov exponential propagator
    
    
    """
    
    def __init__( self, td_density, td_hamiltonian, td_overlap, solver, 
                  preconditioner, degree, gd, timer ):
        """Create SemiImplicitKrylovExponential-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            the time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            the time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner
        degree: integer
            Degree of the Krylov subspace
        gd: GridDescriptor
            coarse (wavefunction) grid descriptor
        timer: Timer
            timer
        
        """
        DummyPropagator.__init__(self, td_density, td_hamiltonian, td_overlap,
                                solver, preconditioner, gd, timer)
        self.kdim = degree + 1
        
        self.tmp_kpt_u = None
        self.lm = None
        self.em = None
        self.hm = None
        self.sm = None
        self.xm = None
        self.qm = None
        self.Hqm = None
        self.Sqm = None
        self.rqm = None
        
        
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions once.
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points
        time: float
            the current time
        time_step: float
            time step
        """

        self.niter = 0
        # temporary wavefunctions
        if self.tmp_kpt_u is None:
            self.tmp_kpt_u = []
            for kpt in kpt_u:
                tmp_kpt = DummyKPoint()
                tmp_kpt.psit_nG = self.gd.empty( n=len(kpt.psit_nG),
                                                 dtype=complex )
                self.tmp_kpt_u.append(tmp_kpt)


        # Allocate memory for Krylov subspace stuff
        nvec = len(kpt_u[0].psit_nG)

        # em = (wfs, degree)
        if self.em is None:
            self.em = npy.zeros( (nvec, self.kdim), float)

        # lm = (wfs)
        if self.lm is None:
            self.lm = npy.zeros( (nvec,), complex)

        # hm = (wfs, degree, degree)
        if self.hm is None:
            self.hm = npy.zeros( (nvec, self.kdim, self.kdim), complex)
        # sm = (wfs, degree, degree)
        if self.sm is None:
            self.sm = npy.zeros( (nvec, self.kdim, self.kdim), complex)
        # xm = (wfs, degree, degree)
        if self.xm is None:
            self.xm = npy.zeros( (nvec, self.kdim, self.kdim), complex)

        # qm = (degree, wfs, nx, ny, nz) 
        if self.qm is None:
            self.qm = self.gd.zeros( (self.kdim, nvec), 
                                     dtype=complex )
        # H qm = (degree, wfs, nx, ny, nz) 
        if self.Hqm is None:
            self.Hqm = self.gd.zeros( (self.kdim, nvec), 
                                     dtype=complex )
        # S qm = (degree, wfs, nx, ny, nz) 
        if self.Sqm is None:
            self.Sqm = self.gd.zeros( (self.kdim, nvec), 
                                     dtype=complex )
        # rqm = (wfs, nx, ny, nz) 
        if self.rqm is None:
            self.rqm = self.gd.zeros( (nvec,), 
                                      dtype=complex )

        
        #self.time_step = time_step #TODO! verify that commenting this out is dependency-safe


        self.timer.start('Update time-dependent operators')

        # rho(t)
        self.td_density.update()
        # H(t)
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        # S(t)
        self.td_overlap.update()

        self.timer.stop('Update time-dependent operators')


        # copy current wavefunctions to temporary variable
        for u in range(len(kpt_u)):
            self.tmp_kpt_u[u].psit_nG[:] = kpt_u[u].psit_nG

        # predict
        if self.debug: print '\nSIKE: Predictor step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        self.solve_propagation_equation(kpt_u, time_step)


        self.timer.start('Update time-dependent operators')

        # rho(t+dt)
        self.td_density.update()
        # H(t+dt/2)
        self.td_hamiltonian.half_update( self.td_density.get_density(),
                                         time + time_step )
        # S(t+dt/2)
        self.td_overlap.half_update()

        self.timer.stop('Update time-dependent operators')


        # propagate psit(t), not psit(t+dt), in correct
        for u in range(len(kpt_u)):
            kpt_u[u].psit_nG[:] = self.tmp_kpt_u[u].psit_nG

        # correct
        if self.debug: print '\nSIKE: Corrector step for psit(t+dt) with t=%g, dt=%g' % (time,time_step)
        self.solve_propagation_equation(kpt_u, time_step)
        
        return self.niter
    
    # psi(t+dt) = exp(-i dt S^-1 H) psi(t)
    def solve_propagation_equation(self, kpt_u, time_step): #TODO! for each kpt individually... how?
        nvec = len(kpt_u[0].psit_nG)
        tmp = npy.zeros((nvec,), complex)
        xm_tmp = npy.zeros((nvec, self.kdim), complex)
        qm = self.qm
        Hqm = self.Hqm
        Sqm = self.Sqm

        # for each kpt_u
        for kpt in kpt_u:
            # Information needed by solver.solve -> self.dot
            self.kpt = kpt
            self.time_step = time_step

            scale = self.create_krylov_subspace( kpt,
                                                 self.td_hamiltonian,
                                                 self.td_overlap,
                                                 self.qm, self.Hqm, self.Sqm )

            # Calculate hm and sm
            for i in range(self.kdim):
                for j in range(self.kdim):
                    self.mblas.multi_zdotc(tmp, qm[i], Hqm[j], nvec)
                    tmp *= self.gd.dv
                    for k in range(nvec):
                        self.hm[k][i][j] = tmp[k]
                    self.mblas.multi_zdotc(tmp, qm[i], Sqm[j], nvec)
                    tmp *= self.gd.dv
                    for k in range(nvec):
                        self.sm[k][i][j] = tmp[k]

            #print 'Hm ='
            #print npy.round(self.hm*1e4) / 1e4
            #print 'log Hm ='
            #print npy.round(npy.log(self.hm)*1e2)/1e2
            #print 'Sm ='
            #print npy.round(self.sm*1e4) / 1e4

            #print npy.round(npy.log10(npy.abs(npy.linalg.eigh(self.hm[0])[1]))*1e6)/1e6

            # Diagonalize
            # Propagate
            # psi(t) = Qm Xm exp(-i Em t) Xm^H Sm e_1
            #        = Qm Xm exp(-i Em t) Sm Qm^H S psi(0) ???
            #        = Qm Xm exp(-i Em t) y
            #        = Qm Xm z
            # y = Sm Qm^H S psi(0) = Xm^H Sm e_1
            # if Sm = I then y is the first row of Xm^*
            # and z = exp(-i Em t) y
            for k in range(nvec):
                (self.em[k], self.xm[k]) = npy.linalg.eigh(self.hm[k])
            #print 'Em = ', self.em
            #for k in range(nvec):
                #print 'Xm',k,' = '
                #print self.xm[k]

            #print self.em[0] * (-1.0J*time_step)
            self.em = npy.exp(self.em * (-1.0J*time_step))
            #print self.em[0]
            #print npy.linalg.eigh(self.hm[0])
            for k in range(nvec):
                z = self.em[k] * npy.conj(self.xm[k,0])
                xm_tmp[k][:] = npy.dot(self.xm[k], z)
            #print xm_tmp
            kpt.psit_nG[:] = 0.0
            for k in range(nvec):
                for i in range(self.kdim):
                    #print 'Xm_tmp[',k,'][',i,'] = ', xm_tmp[k][i]
                    axpy( xm_tmp[k][i] / scale[k], 
                          self.qm[i][k], kpt.psit_nG[k] )

            #print self.qm
            #print kpt.psit_nG


    # Create Krylov subspace
    #    K_v = { psi, S^-1 H psi, (S^-1 H)^2 psi, ... }
    def create_krylov_subspace(self, kpt, h, s, qm, Hqm, Sqm):
        nvec = len(kpt.psit_nG)
        # tmp = (wfs)
        tmp = npy.zeros( (nvec,), complex)
        scale = npy.zeros( (nvec,), complex)
        scale[:] = 0.0
        rqm = self.rqm

        # q_0 = psi
        rqm[:] = kpt.psit_nG

        for i in range(self.kdim):
            qm[i][:] = rqm

            # S orthogonalize
            # q_i = q_i - sum_j<i <q_j|S|q_i> q_j
            for j in range(i):
                self.mblas.multi_zdotc(tmp, qm[i], Sqm[j], nvec)
                tmp *= self.gd.dv
                tmp = npy.conj(tmp)
                self.mblas.multi_zaxpy(-tmp, qm[j], qm[i], nvec)

            # S q_i
            s.apply(kpt, qm[i], Sqm[i])
            self.mblas.multi_zdotc(tmp, qm[i], Sqm[i], nvec)
            tmp *= self.gd.dv
            self.mblas.multi_scale(1./npy.sqrt(tmp), qm[i], nvec)
            self.mblas.multi_scale(1./npy.sqrt(tmp), Sqm[i], nvec)
            if i == 0:
                scale[:] = 1/npy.sqrt(tmp)
                #print 'Scale', scale

            # H q_i
            h.apply(kpt, qm[i], Hqm[i])

            # S r = H q_i, (if stuff, to save one inversion)
            if i+1 < self.kdim:
                rqm[:] = Hqm[i]
                self.solver.solve(self, rqm, Hqm[i])
                #print 'Linear solver iterations = ', self.solver.iterations

        #print '---'
        return scale

    def dot(self, psit, spsit):
        self.td_overlap.apply(self.kpt, psit, spsit)


    ### Below this, just for testing & debug
    def Sdot(self, psit, spsit):
        self.apply_preconditioner(psit, self.tmp)
        self.td_overlap.apply(self.kpt, self.tmp, spsit)

    def Hdot(self, psit, spsit):
        self.apply_preconditioner(psit, self.tmp)
        self.td_hamiltonian.apply(self.kpt, self.tmp, spsit)

    def inverse_overlap(self, kpt_u, degree):
        self.dot = self.Sdot
        self.kpt = kpt_u[0]
        nvec = len(self.kpt.psit_nG)
        nrm2 = npy.zeros( nvec, complex )
        self.tmp = self.gd.zeros( n=nvec, dtype=complex )

        for i in range(10):
            self.solver.solve(self, self.kpt.psit_nG, self.kpt.psit_nG)
            self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.kpt.psit_nG, nvec)
            nrm2 *= self.gd.dv
            self.mblas.multi_scale(1/npy.sqrt(nrm2), self.kpt.psit_nG, nvec)
        self.td_overlap.apply(self.kpt, self.kpt.psit_nG, self.tmp)
        self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.tmp, nvec)
        nrm2 *= self.gd.dv
        print 'S min eig = ', nrm2


    def overlap(self, kpt_u, degree):
        self.dot = self.Sdot
        self.kpt = kpt_u[0]
        nvec = len(self.kpt.psit_nG)
        nrm2 = npy.zeros( nvec, complex )
        self.tmp = self.gd.zeros( n=nvec, dtype=complex )

        for i in range(100):
            self.tmp[:] = self.kpt.psit_nG
            self.td_overlap.apply(self.kpt, self.tmp, self.kpt.psit_nG)
            self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.kpt.psit_nG, nvec)
            nrm2 *= self.gd.dv
            self.mblas.multi_scale(1/npy.sqrt(nrm2), self.kpt.psit_nG, nvec)
        self.td_overlap.apply(self.kpt, self.kpt.psit_nG, self.tmp)
        self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.tmp, nvec)
        nrm2 *= self.gd.dv
        print 'S max eig = ', nrm2


    def inverse_hamiltonian(self, kpt_u, degree):
        self.dot = self.Hdot
        self.kpt = kpt_u[0]
        nvec = len(self.kpt.psit_nG)
        nrm2 = npy.zeros( nvec, complex )
        self.tmp = self.gd.zeros( n=nvec, dtype=complex )

        for i in range(10):
            self.solver.solve(self, self.kpt.psit_nG, self.kpt.psit_nG)
            self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.kpt.psit_nG, nvec)
            nrm2 *= self.gd.dv
            self.mblas.multi_scale(1/npy.sqrt(nrm2), self.kpt.psit_nG, nvec)
        self.td_hamiltonian.apply(self.kpt, self.kpt.psit_nG, self.tmp)
        self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.tmp, nvec)
        nrm2 *= self.gd.dv
        print 'H min eig = ', nrm2


    def hamiltonian(self, kpt_u, degree):
        self.dot = self.Hdot
        self.kpt = kpt_u[0]
        nvec = len(self.kpt.psit_nG)
        nrm2 = npy.zeros( nvec, complex )
        self.tmp = self.gd.zeros( n=nvec, dtype=complex )

        for i in range(100):
            self.tmp[:] = self.kpt.psit_nG
            self.td_hamiltonian.apply(self.kpt, self.tmp, self.kpt.psit_nG)
            self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.kpt.psit_nG, nvec)
            nrm2 *= self.gd.dv
            self.mblas.multi_scale(1/npy.sqrt(nrm2), self.kpt.psit_nG, nvec)
        self.td_hamiltonian.apply(self.kpt, self.kpt.psit_nG, self.tmp)
        self.mblas.multi_zdotc(nrm2, self.kpt.psit_nG, self.tmp, nvec)
        nrm2 *= self.gd.dv
        print 'H max eig = ', nrm2
