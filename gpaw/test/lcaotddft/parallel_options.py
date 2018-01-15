import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

name = 'NaCl'

# Atoms
atoms = molecule(name)
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=6, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            txt='%s_gs.out' % name)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('%s_gs.gpw' % name, mode='all')

# Reference time-propagation calculation
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    txt='%s_td.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)

# Check dipole moment file
world.barrier()
ref = np.loadtxt('%s_dm.dat' % name).ravel()

# Test parallelization options
par_i = []

if world.size == 2:
    par_i.append({'band': 2})
    par_i.append({'sl_default': (2, 1, 2)})
    par_i.append({'sl_default': (1, 2, 4), 'band': 2})
elif world.size == 4:
    par_i.append({'band': 2})
    par_i.append({'sl_default': (2, 2, 2)})
    par_i.append({'sl_default': (2, 2, 4), 'band': 2})
else:
    par_i.append({'band': 2})
    par_i.append({'sl_auto': True})
    par_i.append({'sl_auto': True, 'band': 2})

for i, par in enumerate(par_i):
    td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                        parallel=par,
                        txt='%s_td%d.out' % (name, i))
    DipoleMomentWriter(td_calc, '%s_dm%d.dat' % (name, i))
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)

    world.barrier()
    data = np.loadtxt('%s_dm%d.dat' % (name, i)).ravel()

    tol = 1e-11
    equal(data, ref, tol)
