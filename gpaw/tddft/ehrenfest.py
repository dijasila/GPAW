import numpy as np
from ase.units import Bohr, AUT, _me, _amu
from gpaw.tddft.units import attosec_to_autime
from gpaw.forces import calculate_forces
from gpaw.utilities.scalapack import scalapack_tri2full

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


class EhrenfestVelocityVerlet:
    
    def __init__(self, calc, mass_scale=1.0, setups='paw'):
        """Initializes the Ehrenfest MD calculator.

        Parameters
        ----------

        calc: TDDFT or LCAOTDDFT Object

        mass_scale: 1.0
            Scaling coefficient for atomic masses

        setups: {'paw', 'hgh'}
            Type of setups to use for the calculation

        Note
        ------

        Use propagator = 'EFSICN' for when creating the TDDFT object from a
        PAW ground state calculator and propagator = 'EFSICN_HGH' for HGH
        pseudopotentials and propagator = 'edsicn' for LCAOTDDFT object.
        
        """
        self.calc = calc
        self.setups = setups
        self.positions = self.calc.atoms.positions.copy() / Bohr
        self.positions_new = np.empty_like(self.positions)
        self.velocities = np.empty_like(self.positions)
        amu_to_aumass = _amu / _me
        if self.calc.atoms.get_velocities() is not None:
            self.velocities = self.calc.atoms.get_velocities() / (Bohr / AUT)
        else:
            self.velocities[:] = 0.0
            self.calc.atoms.set_velocities(self.velocities)
        
        self.velocities_half = np.empty_like(self.velocities)
        self.time = 0.0
        
        self.M = calc.atoms.get_masses() * amu_to_aumass * mass_scale

        self.accelerations = np.empty_like(self.velocities)
        self.accelerations_half = np.empty_like(self.accelerations)
        self.accelerations_new = np.empty_like(self.accelerations)
        self.Forces = np.empty_like(self.accelerations)

        self.calc.get_td_energy()
        self.Forces = self.get_forces()

        for i in range(len(self.Forces)):
            self.accelerations[i] = self.Forces[i] / self.M[i]
        if self.calc.wfs.mode == 'lcao':
            self.calc.wfs.Ehrenfest_force_flag = calc.Ehrenfest_force_flag
            self.calc.wfs.S_flag = calc.S_flag
            self.calc.td_hamiltonian.PLCAO_flag = calc.PLCAO_flag
            self.velocities = np.empty_like(self.positions)

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
        self.positions = self.calc.atoms.positions.copy() / Bohr
        self.velocities = self.calc.atoms.get_velocities() / (Bohr / AUT)

        dt = dt * attosec_to_autime
        # save S
        if self.calc.wfs.mode == 'lcao':
            ksl = self.calc.wfs.ksl
            self.using_blacs = ksl.using_blacs
            using_blacs = ksl.using_blacs
            if self.using_blacs:
                self.get_full_overlap()
            self.calc.save_old_S_MM()

        # m a(t+dt)   = F[psi(t),x(t)]
        self.calc.atoms.positions = self.positions * Bohr
        self.calc.set_positions(self.calc.atoms)

        if self.calc.wfs.mode == 'lcao':
            self.move_basis(dt)

        self.calc.get_td_energy()

        if self.calc.wfs.mode == 'lcao' and \
                self.calc.wfs.Ehrenfest_force_flag is True:
            self.calc.get_F_EC()
            self.Forces = self.get_forces() + self.calc.F_EC
        else:
            self.Forces = self.get_forces()

        for i in range(len(self.Forces)):
            self.accelerations[i] = self.Forces[i] / self.M[i]

        # x(t+dt/2)   = x(t) + v(t) dt/2 + .5 a(t) (dt/2)^2
        # vh(t+dt/2)  = v(t) + .5 a(t) dt/2
        self.positions_half = self.positions + self.velocities * \
            dt / 2 + .5 * self.accelerations * dt / 2 * dt / 2
        self.velocities_quarter = self.velocities + .5 * \
            self.accelerations * dt / 2
  
        # m a(t+dt/2) = F[psi(t),x(t+dt/2)a]
        self.calc.atoms.positions = self.positions_half * Bohr
        self.calc.set_positions(self.calc.atoms)

        if self.calc.wfs.mode == 'lcao':
            self.move_basis(dt)

        self.calc.get_td_energy()

        if self.calc.wfs.mode == 'lcao' and \
                self.calc.wfs.Ehrenfest_force_flag is True:
            self.calc.get_F_EC()
            self.Forces = self.get_forces() + self.calc.F_EC
        else:
            self.Forces = self.get_forces()

        for i in range(len(self.Forces)):
            self.accelerations_half[i] = self.Forces[i] / self.M[i]

        # v(t+dt/2)   = vh(t+dt/2) + .5 a(t+dt/2) dt/2
        self.velocities_half = self.velocities_quarter + 0.5 * \
            self.accelerations_half * dt / 2

        # Propagate wf
        # psi(t+dt)   = U(t,t+dt) psi(t)
        self.propagate_single(dt)

        # m a(t+dt/2) = F[psi(t+dt),x(t+dt/2)]
        self.calc.atoms.positions = self.positions_half * Bohr
        self.calc.set_positions(self.calc.atoms)

        if self.calc.wfs.mode == 'lcao':
            self.move_basis(dt)

        self.calc.get_td_energy()
        if self.calc.wfs.mode == 'lcao' and \
                self.calc.wfs.Ehrenfest_force_flag is True:
            self.calc.get_F_EC()
            self.Forces = self.get_forces() + self.calc.F_EC
        else:
            self.Forces = self.get_forces()

        for i in range(len(self.Forces)):
            self.accelerations_half[i] = self.Forces[i] / self.M[i]

        # x(t+dt)     = x(t+dt/2) + v(t+dt/2) dt/2 + .5 a(t+dt/2) (dt/2)^2
        # vh(t+dt)    = v(t+dt/2) + .5 a(t+dt/2) dt/2
        self.positions_new = self.positions_half + self.velocities_half * \
            dt / 2 + 0.5 * self.accelerations_half * dt / 2 * dt / 2
        self.velocities_quarter = self.velocities_half + .5 * \
            self.accelerations_half * dt / 2

        # m a(t+dt)   = F[psi(t+dt),x(t+dt)]
        self.calc.atoms.positions = self.positions_new * Bohr
        self.calc.set_positions(self.calc.atoms)

        if self.calc.wfs.mode == 'lcao':
            self.move_basis(dt)

        self.calc.get_td_energy()
        self.calc.update_eigenvalues()

        if self.calc.wfs.mode == 'lcao' and \
                self.calc.wfs.Ehrenfest_force_flag is True:
            self.calc.get_F_EC()
            self.Forces = self.get_forces() + self.calc.F_EC
        else:
            self.Forces = self.get_forces()

        for i in range(len(self.Forces)):
            self.accelerations_new[i] = self.Forces[i] / self.M[i]

        # v(t+dt)     = vh(t+dt/2) + .5 a(t+dt/2) dt/2
        self.velocities_new = self.velocities_quarter + .5 * \
            self.accelerations_new * dt / 2

        # update
        self.positions[:] = self.positions_new
        self.velocities[:] = self.velocities_new
        self.accelerations[:] = self.accelerations_new

        if self.calc.wfs.mode == 'lcao':
            self.calc.timer.start('BASIS CHANGE')
            if self.calc.S_flag is True:
                if using_blacs:
                    self.get_full_overlap()
                # Change basis when atoms move
                self.calc.basis_change(self.time, dt)
            self.calc.timer.stop('BASIS CHANGE')

        # update atoms
        self.calc.atoms.set_positions(self.positions * Bohr)
        self.calc.atoms.set_velocities(self.velocities * Bohr / AUT)

    def get_full_overlap(self):
        ksl = self.calc.wfs.ksl
        self.mm_block_descriptor = ksl.mmdescriptor
        for kpt in self.calc.wfs.kpt_u:
            scalapack_tri2full(self.mm_block_descriptor, kpt.S_MM)
            scalapack_tri2full(self.mm_block_descriptor, kpt.T_MM)

    def propagate_single(self, dt):
        if self.setups == 'paw':
            self.calc.propagator.propagate(self.time, dt, self.velocities_half)
        else:
            self.calc.propagator.propagate(self.time, dt)

    def get_energy(self):
        """Updates kinetic, electronic and total energies"""

        self.Ekin = 0.5 * (self.M * (self.velocities**2).sum(axis=1)).sum()
        self.e_coulomb = self.calc.get_td_energy()
        self.Etot = self.Ekin + self.e_coulomb
        return self.Etot
        
    def get_velocities_in_au(self):
        return self.velocities

    def set_velocities_in_au(self, v):
        self.velocities[:] = v
        self.calc.atoms.set_velocities(v * Bohr / AUT)

    def move_basis(self, dt):
        self.calc.timer.start('BASIS CHANGE')
        if self.calc.S_flag is True:
            if self.using_blacs:
                self.get_full_overlap()
            # Change basis when atoms move
            self.calc.basis_change(self.time, dt)
        self.calc.timer.stop('BASIS CHANGE')
        ksl = self.calc.wfs.ksl
        self.using_blacs = ksl.using_blacs
        if self.using_blacs:
            self.get_full_overlap()
        self.calc.save_old_S_MM()
