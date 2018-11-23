import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('SiH4')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=7, h=0.4,
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            xc='GLLBSC',
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)

# Write a restart point
td_calc.write('td.gpw', mode='all')

# Keep propagating
td_calc.propagate(20, 3)

# Restart from the restart point
td_calc = LCAOTDDFT('td.gpw', txt='td2.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.propagate(20, 3)
world.barrier()

# Check dipole moment file
data_tj = np.loadtxt('dm.dat')
# Original run
ref_i = data_tj[4:8].ravel()
# Restarted steps
data_i = data_tj[8:].ravel()

tol = 1e-10
equal(data_i, ref_i, tol)

# Test the absolute values
data = np.loadtxt('dm.dat')[:8].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.000000000000e+00,
       2.437239510000e-15,
       -3.417949849296e-14,
       -1.289432868020e-14,
       -1.612459943472e-14,
       0.000000000000e+00,
       1.767648140000e-15,
       -3.290651055510e-14,
       1.482125109544e-14,
       -1.067957790964e-14,
       8.268274700000e-01,
       -2.106985000000e-15,
       6.206570044186e-05,
       6.206570048250e-05,
       6.206570045713e-05,
       1.653654930000e+00,
       -5.197858830000e-16,
       1.002339816037e-04,
       1.002339816474e-04,
       1.002339816283e-04,
       2.480482400000e+00,
       -3.619014160000e-15,
       1.071096524230e-04,
       1.071096524562e-04,
       1.071096524337e-04,
       3.307309870000e+00,
       5.417301690000e-16,
       9.215481351756e-05,
       9.215481353143e-05,
       9.215481352838e-05,
       4.134137330000e+00,
       -1.327424650000e-15,
       6.846554728850e-05,
       6.846554727538e-05,
       6.846554727659e-05,
       4.960964800000e+00,
       -2.094576110000e-15,
       4.180335038167e-05,
       4.180335035045e-05,
       4.180335036931e-05]

tol = 1e-12
equal(data, ref, tol)
