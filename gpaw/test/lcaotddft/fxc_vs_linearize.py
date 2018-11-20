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

fxc = 'LDA'
# Time-propagation calculation with fxc
td_calc = LCAOTDDFT('gs.gpw', fxc=fxc, txt='td_fxc.out')
DipoleMomentWriter(td_calc, 'dm_fxc.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4)

# Time-propagation calculation with linearize_to_fxc()
td_calc = LCAOTDDFT('gs.gpw', txt='td_lin.out')
td_calc.linearize_to_xc(fxc)
DipoleMomentWriter(td_calc, 'dm_lin.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4)

# Test the equivalence
world.barrier()
ref = np.loadtxt('dm_fxc.dat').ravel()
data = np.loadtxt('dm_lin.dat').ravel()

tol = 1e-9
equal(data, ref, tol)

# Test the absolute values
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.000000000000e+00,
       -2.220446050000e-15,
       -4.440892098501e-15,
       2.398081733190e-14,
       2.753353101070e-14,
       0.000000000000e+00,
       -6.661338150000e-16,
       -1.953992523340e-14,
       5.329070518201e-15,
       5.329070518201e-15,
       8.268274700000e-01,
       -1.110223020000e-15,
       6.179473583101e-05,
       6.179473585544e-05,
       6.179473585100e-05,
       1.653654930000e+00,
       2.275957200000e-15,
       9.897615335674e-05,
       9.897615338161e-05,
       9.897615336918e-05,
       2.480482400000e+00,
       -3.885780590000e-16,
       1.043950212098e-04,
       1.043950212241e-04,
       1.043950212054e-04,
       3.307309870000e+00,
       -1.720845690000e-15,
       8.855677772202e-05,
       8.855677773134e-05,
       8.855677772068e-05]

tol = 1e-12
equal(data, ref, tol)
