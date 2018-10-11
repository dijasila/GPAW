from __future__ import print_function
import numpy as np
from gpaw.pgs import GPAWULMSymmetryCalculator
from gpaw.pgs import tools
import gpaw.mpi
from gpaw import GPAW
from ase.build import molecule


# Ground state:

a = 8.0
h = 0.2

name = 'H2O'
system = molecule(name)
system.set_cell((a, a, a))
system.center()

calc = GPAW(h=h,
            nbands=8,
            txt='%s-gpaw.txt' % name
            )
system.set_calculator(calc)
e = system.get_potential_energy()

calc.write('%s.gpw' % name, mode='all')


# Symmetry analysis:

symcalc = GPAWULMSymmetryCalculator(filename='H2O.gpw',
                                    statelist=range(4),
                                    pointgroup='C2v',
                                    mpi=gpaw.mpi,
                                    overlapfile='overlaps-h2o.txt',
                                    symmetryfile='symmetries-h2o.txt')

symcalc.initialize()

# Define atom indices around which the analysis is run:
coreatoms = range(3)

# Define the secondary axis atoms. The component of the vector
# between these atoms parallel to the main axis is rotated toalong
# the x-axis
secaxisatoms = [0, 2]

# Determine the main axis:
mainaxis = (symcalc.atoms[0].position - (symcalc.atoms[1].position + 
                                         symcalc.atoms[2].position) / 2.)

# Determine the secondary axis:
secaxis = tools.get_axis(symcalc.atoms, secaxisatoms)

# Calculate and deliver the required rotations:
Rx, Ry, Rz =  tools.get_rotations_for_axes(mainaxis, secaxis)
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
                                            cutlimit=3.0))

# Set up the shift vector based on the center-of-mass of `coreatoms`:
symcalc.set_shiftvector(tools.calculate_shiftvector(atoms=symcalc.atoms,
                                                    coreatoms=coreatoms,
                                                    gridspacing=h))

# Print the pre-analysis transformations:
if gpaw.mpi.rank == 0:
    print('Shift vector (in angstroms): ', symcalc.shiftvector)
    print('Rotations: rotate x by %.06f; rotate y by %.06f; rotate z by %.06f'
          % (Rx, -Ry, Rz))

# Calculate the symmetry representation weights of the wave functions:
symcalc.calculate(analyze=True)
