from gpaw.pgs import GPAWULMSymmetryCalculator
from gpaw.pgs import tools

import gpaw.mpi
from gpaw import GPAW
from ase.build import molecule

from gpaw.test import equal
from ase import Atom, Atoms
import math
import numpy as np

deg2rad = 2 * math.pi / 360.



# Ground state:
name = 'Mg-H2O-6'
MgO = 2.09 #Mg-O bond length

atoms = Atoms()
atoms.append(Atom('Mg', position=[0, 0, 0]))

h2o = molecule('H2O')
opos = h2o[0].position.copy()
for atom in h2o: 
    atom.position -= opos

for atom in h2o:
    atom.position += np.array([0., 0., -MgO])

for atom in h2o:
    atoms.append(atom)
    atom.position *= -1
    atoms.append(atom)

for atom in h2o:
    atom.position = np.array([atom.position[1], 
                              atom.position[0], 
                              atom.position[2]])
    atom.position = np.array([atom.position[2],
                              atom.position[1],
                              atom.position[0]])
    atoms.append(atom)
    atom.position *= -1
    atoms.append(atom)

for atom in h2o:
    atom.position = np.array([atom.position[0], 
                              atom.position[2], 
                              atom.position[1]])
    atom.position = np.array([atom.position[1],
                              atom.position[0],
                              atom.position[2]])
    atoms.append(atom)
    atom.position *= -1
    atoms.append(atom)
system = atoms

system.center(vacuum=5.0)

h = 0.2
calc = GPAW(h=h,
            nbands=40,
            charge=2,
            txt='%s-gpaw.txt' % name
            )
system.set_calculator(calc)
e = system.get_potential_energy()

calc.write('%s.gpw' % name, mode='all')


# Symmetry analysis:

symcalc = GPAWULMSymmetryCalculator(filename='%s.gpw' % name,
                                    statelist=range(25),
                                    pointgroup='Th',
                                    mpi=gpaw.mpi,
                                    overlapfile='overlaps_%s.txt' % name,
                                    symmetryfile='symmetries_%s.txt' % name)

symcalc.initialize()

coreatoms = range(len(symcalc.atoms))

P = symcalc.atoms.positions


# C3 axis is the main axis
mainaxis = (P[2] + P[7] + P[14])/3. - P[0] #from Mg to center of 3 O atoms 
secondaryaxis = P[7] - P[0] #from Mg to one of O atoms

# Deliver the required rotations:
Rx, Ry, Rz = tools.get_rotations_for_axes(mainaxis, secondaryaxis)
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

