from ase.units import Bohr, AUT, _me, _amu
from gpaw.tddft.units import attosec_to_autime
from gpaw.forces import calculate_forces
import numpy as np

###############################################################################
# EHRENFEST DYNAMICS WITHIN THE PAW METHOD
# WORKS WITH PAW AS LONG THERE ISN'T TOO
# TOO MUCH OVERLAP BETWEEN THE SPHERES
# SUPPORTS ALSO HGH PSEUDOPOTENTIALS
###############################################################################

# m a(t+dt)   = F[psi(t),x(t)]
# x(t+dt/2)   = x(t) + v(t) dt/2 + .5 a(t) (dt/2)^2
# vh(t+dt/2)  = v(t) + .5 a(t) dt/2
# m a(t+dt/2) = F[psi(t),x(t+dt/2)]
# v(t+dt/2)   = vh(t+dt/2) + .5 a(t+dt/2) dt/2
#
# psi(t+dt)   = U(t,t+dt) psi(t)
#
# m a(t+dt/2) = F[psi(t+dt),x(t+dt/2)]
# x(t+dt)     = x(t+dt/2) + v(t+dt/2) dt/2 + .5 a(t+dt/2) (dt/2)^2
# vh(t+dt)    = v(t+dt/2) + .5 a(t+dt/2) dt/2
# m a(t+dt)   = F[psi(t+dt),x(t+dt)]
# v(t+dt)     = vh(t+dt/2) + .5 a(t+dt/2) dt/2

# TODO: move force corrections from forces.py to this module, as well as
# the cg method for calculating the inverse of S from overlap.py


class EhrenfestVelocityVerletLCAO:
    
    def __init__(self, calc, mass_scale=1.0, setups='paw'):
        """Initializes the Ehrenfest MD LCAO calculator.

        Parameters
        ----------

        calc: LCAOTDDFT Object

        mass_scale: 1.0
            Scaling coefficient for atomic masses

        setups: {'paw', 'hgh'}
            Type of setups to use for the calculation

        Note
        ------

        Use propagator = 'SICN' for when creating the LCAOTDDFT object from a
        PAW ground state calculator.
        
        """
        self.calc = calc
        self.calc.wfs.ED_F = calc.ED_F
        self.calc.td_hamiltonian.P_flag = calc.PP_flag
        self.calc.wfs.calculatePD = self.calc.td_hamiltonian.PPP.calc_P
        self.setups = setups
        self.x = self.calc.atoms.positions.copy() / Bohr
        self.xn = self.x.copy()
        self.v = self.x.copy()
        self.calc.wfs.v = self.v
        amu_to_aumass = _amu / _me
        if self.calc.atoms.get_velocities() is not None:
            self.v = self.calc.atoms.get_velocities() / (Bohr / AUT)
        else:
            self.v[:] = 0.0
            self.calc.atoms.set_velocities(self.v)
        
        self.vt = self.v.copy()
        self.vh = self.v.copy()
        self.time = 0.0
        
        self.M = calc.atoms.get_masses() * amu_to_aumass * mass_scale

        self.a = self.v.copy()
        self.ah = self.a.copy()
        self.an = self.a.copy()
        self.F = self.a.copy()

        self.calc.get_td_energy()    # Need modification
        self.F = self.get_forces()

        for kpt in self.calc.wfs.kpt_u:
            self.S_MM_old=np.empty_like(kpt.S_MM)
    
        for i in range(len(self.F)):
            self.a[i] = self.F[i] / self.M[i]

    def get_forces(self):
        return calculate_forces(self.calc.wfs,
                                self.calc.td_density.get_density(),
                                self.calc.td_hamiltonian.hamiltonian,
                                self.calc.log)

    def propagate(self, dt):
        """Performs one Ehrenfest MD propagation step

        Parameters
        ----------

        dt: scalar
            Time step (in attoseconds) used for the Ehrenfest MD step

        """

        self.x = self.calc.atoms.positions.copy() / Bohr
        self.v = self.calc.atoms.get_velocities() / (Bohr / AUT)

        dt = dt * attosec_to_autime
        self.calc.save_old_S_MM()
        # m a(t+dt)   = F[psi(t),x(t)]
        self.calc.atoms.positions = self.x * Bohr
        self.calc.set_positions(self.calc.atoms)
        self.calc.get_td_energy() 
        self.F = self.get_forces()
 
        for i in range(len(self.F)):
            self.a[i] = self.F[i] / self.M[i]

        # x(t+dt/2)   = x(t) + v(t) dt/2 + .5 a(t) (dt/2)^2
        # vh(t+dt/2)  = v(t) + .5 a(t) dt/2
        self.xh = self.x + self.v * dt / 2 + .5 * self.a * dt / 2 * dt / 2
        self.vhh = self.v + .5 * self.a * dt / 2

        # m a(t+dt/2) = F[psi(t),x(t+dt/2)a]
        self.calc.atoms.positions = self.xh * Bohr
        self.calc.set_positions(self.calc.atoms)
        self.calc.get_td_energy()
        self.F = self.get_forces()

        for i in range(len(self.F)):
            self.ah[i] = self.F[i] / self.M[i]

        # v(t+dt/2)   = vh(t+dt/2) + .5 a(t+dt/2) dt/2
        self.vh = self.vhh + 0.5 * self.ah * dt / 2

        # Propagate wf
        # psi(t+dt)   = U(t,t+dt) psi(t)
        self.calc.timer.start('PROPAGATE WF')
        self.propagate_single(dt)
        self.calc.timer.stop('PROPAGATE WF')

        # m a(t+dt/2) = F[psi(t+dt),x(t+dt/2)]
        self.calc.atoms.positions = self.xh * Bohr
        self.calc.set_positions(self.calc.atoms)
        self.calc.get_td_energy()
        self.F = self.get_forces()
 
        for i in range(len(self.F)):
            self.ah[i] = self.F[i] / self.M[i]
 
        # x(t+dt)     = x(t+dt/2) + v(t+dt/2) dt/2 + .5 a(t+dt/2) (dt/2)^2
        # vh(t+dt)    = v(t+dt/2) + .5 a(t+dt/2) dt/2
        self.xn = self.xh + self.vh * dt / 2 + 0.5 * self.ah * dt / 2 * dt / 2
        self.vhh = self.vh + .5 * self.ah * dt / 2

        # m a(t+dt)   = F[psi(t+dt),x(t+dt)]
        self.calc.atoms.positions = self.xn * Bohr
        self.calc.set_positions(self.calc.atoms)
        self.calc.get_td_energy()
        self.calc.update_eigenvalues()
        self.F = self.get_forces()

        for i in range(len(self.F)):
            self.an[i] = self.F[i] / self.M[i]

         # v(t+dt)     = vh(t+dt/2) + .5 a(t+dt/2) dt/2
        self.vn = self.vhh + .5 * self.an * dt / 2

        # update
        self.x[:] = self.xn
        self.v[:] = self.vn
        self.a[:] = self.an

        # update atoms
        self.calc.atoms.set_positions(self.x * Bohr)
        self.calc.atoms.set_velocities(self.v * Bohr / AUT)

        self.calc.timer.start('PROPAGATE WF S1/2')
        if self.calc.S_flag == True:
            # propagate LCAO C using overlap matrix
            self.calc.propagate_using_S12(self.time, dt)
        self.calc.timer.stop('PROPAGATE WF S1/2')

        self.calc.set_positions(self.calc.atoms)
        self.calc.get_td_energy()

    def propagate_single(self, dt):

        if self.setups == 'paw':
            self.calc.propagator.propagate(self.time, dt, self.vh)
        else:
            self.calc.propagator.propagate(self.time, dt)

    def get_energy(self):
        """Updates kinetic, electronic and total energies"""

        self.Ekin = 0.5 * (self.M * (self.v**2).sum(axis=1)).sum()
        self.e_coulomb = self.calc.get_td_energy()
        self.Etot = self.Ekin + self.e_coulomb
        return self.Etot
        
    def get_velocities_in_au(self):
        return self.v

    def set_velocities_in_au(self, v):
        self.v[:] = v
        self.calc.atoms.set_velocities(v * Bohr / AUT)
