import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('NaCl')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=6, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            xc='LDA',
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

fxc = 'PBE'
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

# Test
world.barrier()
ref = np.loadtxt('dm_fxc.dat').ravel()
data = np.loadtxt('dm_lin.dat').ravel()

tol = 1e-9
equal(data, ref, tol)
