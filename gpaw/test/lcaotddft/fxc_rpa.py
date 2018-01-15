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
            txt='%s_gs.out' % name)
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('%s_gs.gpw' % name, mode='all')

# Time-propagation calculation with fxc
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    fxc='RPA',
                    txt='%s_td.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)
world.barrier()

# Check dipole moment file
data = np.loadtxt('%s_dm.dat' % name)[:, 2:].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [4.786589735249e-15,
       6.509942495725e-15,
       3.836848815869e-14,
       4.429061708370e-15,
       7.320865686028e-15,
       2.877243538173e-14,
       1.960742479669e-05,
       1.960742479842e-05,
       1.804029540200e-05,
       3.761996854449e-05,
       3.761996854564e-05,
       3.596679132063e-05,
       5.257366852049e-05,
       5.257366852232e-05,
       5.366659968830e-05]

tol = 1e-12
equal(data, ref, tol)
