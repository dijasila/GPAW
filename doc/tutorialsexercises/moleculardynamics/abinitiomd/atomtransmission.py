from ase.lattice.hexagonal import Graphene
from gpaw import GPAW
from ase import Atoms
from ase.io import Trajectory
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from gpaw.utilities import h2gpts

name = 'graphene_h'

# 5 x 5 supercell of graphene
index1 = 5
index2 = 5
a = 2.45
c = 3.355

gra = Graphene(symbol='C',
               latticeconstant={'a': a, 'c': c},
               size=(index1, index2, 1),
               )

gra.center(vacuum=10.0, axis=2)
gra.center()

# Projectile impact point at the above a hexagon center
projpos = [[gra[15].position[0], gra[15].position[1] + 1.41245, 15.0]]

H = Atoms('H', cell=gra.cell, positions=projpos)
H.mass = 1.0  # Set mass to one atomic mass unit to avoid isotope average

# Combine target and projectile
atoms = gra + H
atoms.set_pbc(True)

calc = GPAW(gpts=h2gpts(0.2, gra.get_cell(), idiv=8),
            nbands=110,
            xc='LDA',
            txt=name + '_gs.txt',
            )

atoms.calc = calc
atoms.get_potential_energy()

# Moving to the MD part
Ekin = 100  # Kinetic energy of the ion (in eV)
timestep = 0.1  # Timestep in fs

# Filename for saving trajectory
ekin_str = '_ek' + str(int(Ekin)) + 'eV'
traj_file = name + ekin_str + '.traj'

proj_idx = 50  # atomic index of the projectile

# Integrator for the equations of motion, timestep depends on system
dyn = VelocityVerlet(atoms, timestep * fs)

# Saving the positions of all atoms after every time step
traj = Trajectory(traj_file, 'w', atoms)
dyn.attach(traj.write, interval=1)

# Giving the target atom a kinetic energy of ene in the -z direction
atoms[proj_idx].momentum[2] = -(2 * Ekin * atoms[proj_idx].mass)**0.5

# Running the simulation for 80 timesteps
dyn.run(80)

traj.close()
