import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

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
            xc='LDA',
            txt='%s_gs.out' % name)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('%s_gs.gpw' % name, mode='all')

# Time-propagation calculation without fxc
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    txt='%s_td.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4)
world.barrier()

# Check dipole moment file
ref = np.loadtxt('%s_dm.dat' % name).ravel()

# Time-propagation calculation with fxc=xc
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    fxc='LDA',
                    txt='%s_td_fxc.out' % name)
DipoleMomentWriter(td_calc, '%s_dm_fxc.dat' % name)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 1)
td_calc.write('%s_td_fxc.gpw' % name, mode='all')

# Restart from the restart point
td_calc = LCAOTDDFT('%s_td_fxc.gpw' % name,
                    txt='%s_td_fxc2.out' % name)
DipoleMomentWriter(td_calc, '%s_dm_fxc.dat' % name)
td_calc.propagate(20, 3)

# Check dipole moment file
data = np.loadtxt('%s_dm_fxc.dat' % name)[[0, 1, 2, 4, 5, 6]].ravel()

tol = 1e-9
equal(data, ref, tol)
