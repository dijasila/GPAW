import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('Na2')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
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
