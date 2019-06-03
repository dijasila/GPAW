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
       1.440746980000e-15,
       -5.150207058975e-14,
       -2.111502433286e-14,
       -7.898943127163e-15,
       0.000000000000e+00,
       2.611197480000e-15,
       -8.396549188150e-14,
       -2.905622138206e-14,
       -3.511635310469e-14,
       8.268274700000e-01,
       2.114669130000e-15,
       6.176327039782e-05,
       6.176327044228e-05,
       6.176327043511e-05,
       1.653654930000e+00,
       -1.592560840000e-15,
       9.885691995366e-05,
       9.885691999260e-05,
       9.885691998996e-05,
       2.480482400000e+00,
       2.175440570000e-17,
       1.041388789639e-04,
       1.041388789863e-04,
       1.041388789903e-04,
       3.307309870000e+00,
       -1.618830990000e-16,
       8.817245095040e-05,
       8.817245095747e-05,
       8.817245094278e-05]

print('result')
print(data.tolist())

tol = 1e-12
equal(data, ref, tol)
