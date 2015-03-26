# ASE provides the structures
from ase import atoms
# and quasinewton finds the lowest energy structure
from ase.optimize import QuasiNewton

# GPAW calculates the properties of a
# given structure (energies, forces, etc.)
from gpaw import GPAW

# Let's make an H2 molecule
# with the experimental bond length
H2 = atoms('H2',
           positions=[(0, 0, 0),
                      (0, 0, 0.74)])
# And use GPAW to calculate energies and forces
H2.set_calculator(GPAW())

# Optimize distance
relax = QuasiNewton(H2)
relax.run()

# Get distance between atom number 0 and 1
d0 = molecule.get_distance(0, 1)

