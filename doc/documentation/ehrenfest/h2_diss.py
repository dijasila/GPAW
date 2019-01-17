from gpaw.tddft import TDDFT
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from gpaw.tddft.laser import CWField
from ase.units import Hartree, Bohr, AUT
from ase.io import Trajectory

name = 'h2_diss'

# Ehrenfest simulation parameters
timestep = 20.0 # Timestep given in attoseconds
ndiv = 10  # Write trajectory every 10 timesteps
niter = 500 # Run for 500 timesteps

# TDDFT calculator with an external potential emulating an intense harmonic laser field
# aligned (CWField uses by default the z axis) along the H2 molecular axis.
tdcalc = TDDFT(name + '_gs.gpw', txt=name + '_td.txt', propagator='EFSICN',
               solver='BiCGStab', td_potential=CWField(1000 * Hartree, 1 * AUT, 10))

# For Ehrenfest dynamics, we use this object for the Velocity Verlet dynamics.
ehrenfest = EhrenfestVelocityVerlet(tdcalc)

# Trajectory to save the dynamics.
traj = Trajectory(name + '_td.traj', 'w', tdcalc.get_atoms())

# Propagates the dynamics for niter timesteps.
for i in range(1, niter + 1):
    ehrenfest.propagate(timestep)

    # Every ndiv timesteps, save an image in the trajectory file.
    if i % ndiv == 0:
        # Currently needed with Ehrenfest dynamics to save energy, forces and velocitites.
        epot = tdcalc.get_td_energy() * Hartree
        F_av = ehrenfest.F * Hartree / Bohr # Forces converted from atomic units
        v_av = ehrenfest.v * Bohr / AUT # Velocities converted from atomic units
        atoms = tdcalc.atoms.copy()
        atoms.set_velocities(v_av) # Needed to save the velocities to the trajectory

        traj.write(atoms, energy=epot, forces=F_av)

traj.close()
