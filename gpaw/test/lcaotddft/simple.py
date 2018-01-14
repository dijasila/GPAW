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

ref_i = [2.235396054492e-15,
         -1.246272775083e-15,
         3.398852375701e-14,
         3.095944586065e-15,
         3.091433191404e-15,
         2.513032652157e-14,
         1.967175496130e-05,
         1.967175495957e-05,
         1.805003680186e-05,
         3.799528916941e-05,
         3.799528917059e-05,
         3.602505573350e-05,
         5.371492026867e-05,
         5.371492026518e-05,
         5.385044939991e-05]

tol = 1e-12
equal(data_i, ref_i, tol)
