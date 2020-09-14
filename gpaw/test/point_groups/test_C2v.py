import numpy as np
from ase.build import molecule

from gpaw.pgs import GPAWULMSymmetryCalculator
from gpaw.pgs import tools
import gpaw.mpi
from gpaw import GPAW
from gpaw.test import equal


def test_C2v(in_tmp_dir):
    a = 8.0
    h = 0.2
    name = 'H2O'

    # Ground state:
    system = molecule(name)
    system.set_cell((a, a, a))
    system.center()

    calc = GPAW(h=h,
                nbands=8,
                txt='%s-gpaw.txt' % name)
    system.set_calculator(calc)
    _ = system.get_potential_energy()

    calc.write('%s.gpw' % name, mode='all')

    # Symmetry analysis:
    symcalc = GPAWULMSymmetryCalculator(
        filename='%s.gpw' % name,
        statelist=range(4),
        pointgroup='C2v',
        mpi=gpaw.mpi,
        overlapfile='overlaps_%s.txt' % name,
        symmetryfile='symmetries_%s.txt' % name)

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
    symcalc.set_shiftvector(
        tools.calculate_shiftvector(
            atoms=symcalc.atoms,
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
            equal(bestweight / norm, 1.0, 0.05)
