from gpaw.pgs import GPAWULMSymmetryCalculator
from gpaw.pgs import tools

import gpaw.mpi
from gpaw import GPAW

from gpaw.test import equal
from ase import Atom, Atoms
import numpy as np

# Ground state:
name = 'allene'
system = Atoms()

CC = 1.33 #C-C bond length
CH = 1.1 #C-H bond length

system.append(Atom('C', position=[0, 0, 0]))
system.append(Atom('C', position=[0, 0, CC]))
system.append(Atom('C', position=[0, 0, -CC]))

c = 1/np.sqrt(2.0)
system.append(Atom('H', position=[0., c*CH, CC + c*CH]))
system.append(Atom('H', position=[0., -c*CH, CC + c*CH]))
system.append(Atom('H', position=[c*CH, 0., -CC - c*CH]))
system.append(Atom('H', position=[-c*CH, 0., -CC - c*CH]))


system.center(vacuum=5.0)

h = 0.2
calc = GPAW(h=h,
            nbands=14,
            charge=0,
            txt='%s-gpaw.txt' % name
            )
system.set_calculator(calc)
e = system.get_potential_energy()

calc.write('%s.gpw' % name, mode='all')


# Symmetry analysis:

symcalc = GPAWULMSymmetryCalculator(filename='%s.gpw'%name,
                                    statelist=range(8),
                                    pointgroup='D2d',
                                    mpi=gpaw.mpi,
                                    overlapfile='overlaps_%s.txt'%name,
                                    symmetryfile='symmetries_%s.txt'%name)

symcalc.initialize()

coreatoms = range(len(symcalc.atoms))

# Deliver the required rotations:
Rx, Ry, Rz = [0., 0., 0.]
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

