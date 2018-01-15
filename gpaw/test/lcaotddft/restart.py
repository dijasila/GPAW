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

# Write a restart point
td_calc.write('%s_td.gpw' % name, mode='all')

# Keep propagating
td_calc.propagate(20, 3)

# Restart from the restart point
td_calc = LCAOTDDFT('%s_td.gpw' % name,
                    txt='%s_td2.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
td_calc.propagate(20, 3)

# Check dipole moment file
data_tj = np.loadtxt('%s_dm.dat' % name)
# Original run
ref_i = data_tj[4:8].ravel()
# Restarted steps
data_i = data_tj[8:].ravel()

tol = 1e-10
equal(data_i, ref_i, tol)
