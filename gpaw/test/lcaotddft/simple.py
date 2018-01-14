import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

from gpaw.test import equal

name = 'Na2'

# Atoms
atoms = molecule(name)
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            txt='%s_gs.out' % name)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('%s_gs.gpw' % name, mode='all')

# Time-propagation calculation
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    txt='%s_td.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)
td_calc.write('%s_td.gpw' % name, mode='all')

# Test dipole moment
data_i = np.loadtxt('%s_dm.dat' % name)[:, 2:].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data_i, 'ref_i', '%.12le')

ref_i = [4.786589735249e-15,
         6.509942495725e-15,
         3.836848815869e-14,
         4.429061708370e-15,
         7.320865686028e-15,
         2.877243538173e-14,
         1.967175332445e-05,
         1.967175332505e-05,
         1.805003047148e-05,
         3.799528613595e-05,
         3.799528613766e-05,
         3.602504333467e-05,
         5.371491630029e-05,
         5.371491629857e-05,
         5.385043148270e-05]

tol = 1e-12
equal(data_i, ref_i, tol)
