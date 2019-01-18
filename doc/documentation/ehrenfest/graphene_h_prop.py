from ase.units import _amu, _me, Bohr, AUT, Hartree
from gpaw import GPAW
from gpaw.tddft import TDDFT
from ase.parallel import paropen
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from ase.io import Trajectory
from gpaw.mpi import world
import numpy as np

name = 'graphene_h'
Ekin = 40e3 # Kinetic energy of the ion (in eV)
timestep = 8.0 * np.sqrt(10e3/Ekin) # Adapted to the ion energy; here 0.5 as (may be too large!)
ekin_str = '_ek' + str(int(Ekin/1000)) + 'k'
amu_to_aumass = _amu/_me
strbody = name + ekin_str
traj_file = strbody + '.traj'

# The parallelization options should match the number of cores, here 8.
p_bands = 2 # Number of bands to parallelise over
dom_dc = (2,2,1) # Domain decomposition for parallelization
parallel = {'band':p_bands, 'domain':dom_dc}

tdcalc = TDDFT(name + '.gpw', propagator='EFSICN', solver='BiCGStab', txt=strbody + '_td.txt', parallel=parallel)

proj_idx = 50 # Atomic index of the projectile
delta_stop = 5.0 / Bohr # Stop condition when ion is within 5 A of cell boundary.

# Setting the initial velocity according to the kinetic energy.
Mproj = tdcalc.atoms.get_masses()[proj_idx] * amu_to_aumass
Ekin *= Mproj / Hartree
v = np.zeros((proj_idx+1,3)) 
v[proj_idx,2] = -np.sqrt((2*Ekin)/Mproj) * Bohr / AUT
tdcalc.atoms.set_velocities(v)

evv = EhrenfestVelocityVerlet(tdcalc)
traj = Trajectory(traj_file, 'w', tdcalc.get_atoms())

trajdiv = 2 # Number of timesteps between trajectory images
densdiv = 10 # Number of timesteps between saved electron densities
niters = 200 # Total number of timesteps to propagate

for i in range(niters):
    # Stopping condition when projectile z coordinate passes threshold
    if evv.x[proj_idx,2] < delta_stop:
        tdcalc.write(strbody + '_end.gpw', mode='all')
        break

    # Saving trajectory file every trajdiv timesteps
    if i % trajdiv == 0:
        F_av = evv.F * Hartree / Bohr # Forces converted from atomic units
        v_av = evv.v * Bohr / AUT # Velocities converted from atomic units
        epot = tdcalc.get_td_energy() * Hartree # Energy converted from atomic units
        atoms = tdcalc.get_atoms().copy()
        atoms.set_velocities(v_av)

        traj.write(atoms, energy=epot, forces=F_av)

    # Saving electron density every densdiv timesteps
    if (i != 0 and i % densdiv == 0):
        tdcalc.write(strbody + '_step'+str(i)+'.gpw')
        v[proj_idx,2] = -np.sqrt((2*Ekin)/Mproj) * Bohr / AUT
        tdcalc.atoms.set_velocities(v)

    evv.propagate(timestep)

traj.close()
