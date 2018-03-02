import numpy as np

from ase.build import molecule
from ase.units import Hartree, Bohr
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.laser import GaussianImpulse
from gpaw.external import ConstantElectricField
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.units import as_to_au
from gpaw.mpi import world

from gpaw.test import equal

# Settings
dt = 20.0
N = 10
kick_v = np.ones(3) * 1e-5

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
td_calc.absorption_kick(kick_v)
td_calc.propagate(dt, N)

# Pulse
direction = kick_v
ext = ConstantElectricField(Hartree / Bohr, direction)
pulse = GaussianImpulse(1e-5, 0e3, 8.6, 0.5, 'sin')

# Time-propagation calculation with pulse
td_calc = LCAOTDDFT('gs.gpw', td_potential={'ext': ext, 'laser': pulse},
                    txt='tdpulse.out')
DipoleMomentWriter(td_calc, 'dmpulse.dat')
td_calc.propagate(dt, N)

# Convoluted dipole moment
world.barrier()
time_t = np.arange(0, dt * (N + 0.1), dt) * as_to_au
pulse_t = pulse.strength(time_t)
dm_tv = np.delete(np.loadtxt('dm.dat')[:, 2:], 1, axis=0)
dm_tv /= np.linalg.norm(kick_v)
pulsedm_tv = np.loadtxt('dmpulse.dat')[:, 2:]

tol = 5e-6
for v in range(3):
    pulsedmconv_t = np.convolve(dm_tv[:, v], pulse_t)[:(N + 1)] * dt * as_to_au
    equal(pulsedm_tv[:, v], pulsedmconv_t, tol)
