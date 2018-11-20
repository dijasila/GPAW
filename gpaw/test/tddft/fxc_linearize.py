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

fxc = 'LDA'
# Time-propagation calculation with linearize_to_fxc()
td_calc = TDDFT('gs.gpw', txt='td.out')
td_calc.linearize_to_xc(fxc)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4, 'dm.dat')

# Test the absolute values
data = np.loadtxt('dm.dat').ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.000000000000e+00,
       -2.862284330000e-15,
       2.541700511956e-14,
       1.557884338717e-16,
       2.695009893022e-14,
       8.268274700000e-01,
       -1.212524360000e-15,
       6.114013824165e-05,
       6.114045331494e-05,
       6.114211153991e-05,
       1.653654930000e+00,
       -1.016462630000e-16,
       1.066792190503e-04,
       1.066810114025e-04,
       1.066824538759e-04,
       2.480482400000e+00,
       -3.390683670000e-17,
       1.342245604820e-04,
       1.342341316473e-04,
       1.342344669447e-04]

tol = 1e-12
equal(data, ref, tol)
