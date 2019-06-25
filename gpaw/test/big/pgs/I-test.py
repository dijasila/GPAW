from gpaw.pgs import GPAWULMSymmetryCalculator
from gpaw.pgs import tools

import gpaw.mpi
from gpaw import GPAW
from ase import Atom, Atoms

from gpaw.test import equal
import math
import numpy as np


# Build molecule:
name = 'dodecaborate'

# Icosahedron params:
BB = 1.8 #B-B bond length
BH = 1.2 #B-H bond length
phi = 0.5 * (1. + math.sqrt(5.))
l = 1.0    #streching parameter
P = BB/2.     #edge length parameter
ico = np.array([[0., l, phi], [0.,l,-phi], [0.,-l, phi], [0.,-l, -phi], 
                [l, phi, 0.], [l,-phi,0.], [-l, phi,0.], [-l, -phi,0.],
                [phi, 0., l], [-phi,0.,l], [phi,0., -l], [-phi, 0.,-l]]) * P

system = Atoms()

for corner in ico:
    system.append(Atom('B', position=corner))
    Hpos = corner + BH * corner / np.linalg.norm(corner)
    system.append(Atom('H', position=Hpos))

system.center(vacuum=5.0)

h = 0.2
calc = GPAW(h=h,
            nbands=50,
            charge=-2,
            txt='%s-gpaw.txt' % name
            )
system.set_calculator(calc)
e = system.get_potential_energy()

calc.write('%s.gpw' % name, mode='all')


# Symmetry analysis:

symcalc = GPAWULMSymmetryCalculator(filename='%s.gpw'%name,
                                    statelist=range(24),
                                    pointgroup='I',
                                    mpi=gpaw.mpi,
                                    overlapfile='overlaps_%s.txt'%name,
                                    symmetryfile='symmetries_%s.txt'%name)

symcalc.initialize()

# Define atom indices around which the analysis is run:
coreatoms = range(len(system))

mainaxisatoms = [0,6]
secaxisatoms = [16, 0]

mainaxis = tools.get_axis(symcalc.atoms, mainaxisatoms)
secaxis = tools.get_axis(symcalc.atoms, secaxisatoms)


# Deliver the required rotations:
Rx, Ry, Rz = tools.get_rotations_for_axes(mainaxis, secaxis)
symcalc.set_initialrotations(Rx, Ry, Rz)

# Determine some parameters from the data:
wfshape = tools.get_wfshape(symcalc)

# Give the grid spacing to the symmetry calculator for shifting
# the atoms to center:
h = tools.get_h(symcalc)
symcalc.set_gridspacing(h)

# Set up the volume where the analysis is restricted:
symcalc.set_cutarea(tools.calculate_cutarea(atoms=symcalc.atoms,
                                            coreatoms=coreatoms,
                                            wfshape=wfshape,
                                            gridspacing=h,
                                            cutlimit=3.00))

# Set up the shift vector based on the center-of-mass of `coreatoms`:
symcalc.set_shiftvector(tools.calculate_shiftvector(atoms=symcalc.atoms,
                                                coreatoms=coreatoms,
                                                gridspacing=h))

# Calculate the symmetry representation weights of the wave functions:
symcalc.calculate(analyze=True)

if gpaw.mpi.rank == 0:
    f = open('symmetries_%s.txt' % name, 'r')
    results = []
    for line in f:
        if line.startswith('#'):
            continue
        results.append(line.split()[:-1])
    f.close()

    results = np.array(results).astype(float)
    for i in range(len(results)):
        norm = results[i, 2]
        bestweight = (results[i, 3:]).max()
        equal(bestweight / norm, 1.0, 0.1)

