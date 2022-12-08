import ase.units as units
from ase import Atoms
from ase.build import graphene
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet
from gpaw import GPAW
from gpaw.utilities import h2gpts

name = 'graphene_h'

# 5 x 5 supercell of graphene
a = 2.45
gra = graphene(a=a, size=(5, 5, 1), vacuum=10)
gra.center()

# Starting position of the projectile with an impact point at the
# center of a hexagon.
# Set mass to one atomic mass unit to avoid isotope average.
atoms = gra + Atoms('H', masses=[1.0])
d = a / 3**0.5
atoms.positions[-1] = atoms.positions[22] + (0, d, 5)
atoms.pbc = (True, True, True)

calc = GPAW(gpts=h2gpts(0.2, gra.get_cell(), idiv=8),
            nbands=110,
            xc='LDA',
            txt=f'{name}_gs.txt')

atoms.calc = calc
atoms.get_potential_energy()

# Moving to the MD part
ekin = 100  # kinetic energy of the ion (in eV)
timestep = 0.1  # timestep in fs

# Filename for saving trajectory
ekin_str = '_ek' + str(int(ekin / 1000)) + 'keV'
strbody = name + ekin_str
traj_file = f'{name}_ek_{ekin}.traj'

# Integrator for the equations of motion, timestep depends on system
dyn = VelocityVerlet(atoms, timestep * units.fs)

# Saving the positions of all atoms after every time step
with Trajectory(traj_file, 'w', atoms) as traj:
    dyn.attach(traj.write, interval=1)

    # Running one timestep before impact
    dyn.run(1)

    # Giving the target atom a kinetic energy of ene in the -z direction
    atoms[-1].momentum[2] = -(2 * ekin * atoms[-1].mass)**0.5

    # Running the simulation for 80 timesteps
    dyn.run(80)
