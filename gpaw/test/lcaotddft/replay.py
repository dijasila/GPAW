import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum
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

# Time-propagation calculation
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    txt='%s_td.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
WaveFunctionWriter(td_calc, '%s_wfw.ulm' % name)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)
td_calc.write('%s_td.gpw' % name, mode='all')

# Restart from the restart point
td_calc = LCAOTDDFT('%s_td.gpw' % name,
                    txt='%s_td2.out' % name)
DipoleMomentWriter(td_calc, '%s_dm.dat' % name)
WaveFunctionWriter(td_calc, '%s_wfw.ulm' % name)
td_calc.propagate(20, 3)
td_calc.propagate(20, 3)
td_calc.propagate(10, 3)
photoabsorption_spectrum('%s_dm.dat' % name, '%s_spec.dat' % name)

# Replay
td_calc = LCAOTDDFT('%s_gs.gpw' % name,
                    propagator=dict(name='%s_wfw.ulm' % name,
                                    update='density'),
                    txt='%s_rep.out' % name)
DipoleMomentWriter(td_calc, '%s_dm_rep.dat' % name)
td_calc.autopropagate()
photoabsorption_spectrum('%s_dm_rep.dat' % name, '%s_spec_rep.dat' % name)

world.barrier()

# Check the spectrum files
# Do this instead of dipolemoment files in order to see that the kick
# was also written correctly in replaying
ref_i = np.loadtxt('%s_spec.dat' % name).ravel()
data_i = np.loadtxt('%s_spec_rep.dat' % name).ravel()

tol = 1e-10
equal(data_i, ref_i, tol)
