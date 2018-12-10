import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.tddft import TDDFT
from gpaw.poisson import PoissonSolver
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('SiH4')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=7, h=0.4,
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            xc='GLLBSC',
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = TDDFT('gs.gpw', txt='td.out')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3, 'dm.dat')

# Write a restart point
td_calc.write('td.gpw', mode='all')

# Keep propagating
td_calc.propagate(20, 3, 'dm.dat')

# Restart from the restart point
td_calc = TDDFT('td.gpw', txt='td2.out')
td_calc.propagate(20, 3, 'dm.dat')
world.barrier()

# Check dipole moment file
data_tj = np.loadtxt('dm.dat')
# Original run
ref_i = data_tj[4:6].ravel()
# Restarted steps
data_i = data_tj[7:].ravel()

tol = 1e-10
equal(data_i, ref_i, tol)

# Test the absolute values
data = np.loadtxt('dm.dat')[:6].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.000000000000e+00,
       -2.862284330000e-15,
       2.541700511956e-14,
       1.557884338717e-16,
       2.695009893022e-14,
       8.268274700000e-01,
       -1.961856820000e-15,
       6.114013713007e-05,
       6.114044984245e-05,
       6.114210884970e-05,
       1.653654930000e+00,
       1.091038880000e-15,
       1.073335183148e-04,
       1.073354111024e-04,
       1.073275516396e-04,
       2.480482400000e+00,
       1.668867320000e-15,
       1.354547990643e-04,
       1.354551741863e-04,
       1.354463877181e-04,
       3.307309870000e+00,
       -9.167377070000e-16,
       1.442985892380e-04,
       1.443010168584e-04,
       1.442970193746e-04,
       4.134137330000e+00,
       -1.947252800000e-17,
       1.350652494171e-04,
       1.350666337990e-04,
       1.350678431128e-04]

tol = 1e-7
equal(data, ref, tol)
