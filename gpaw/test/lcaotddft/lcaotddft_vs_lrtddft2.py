import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.mpi import world

from gpaw.test import equal

name = 'Na2'

# Atoms
atoms = molecule(name)
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
            basis='sz(dzp)', mode='lcao', xc='oldLDA',
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
td_calc.absorption_kick([0, 0, 1e-5])
td_calc.propagate(30, 150)
photoabsorption_spectrum('%s_dm.dat' % name, '%s_spec.dat' % name,
                         e_max=10, width=0.5, delta_e=0.1)


# LrTDDFT2 calculation
calc = GPAW('%s_gs.gpw' % name, txt='%s_lr.out' % name)
lr = LrTDDFT2(name, calc, fxc='LDA')
lr.calculate()
lr.get_spectrum('%s_lr_spec.dat' % name, 0, 10.1, 0.1, width=0.5)
world.barrier()

# Scale spectra due to slightly different definitions
spec1_ej = np.loadtxt('%s_spec.dat' % name)
data1_e = spec1_ej[:, 3]
data1_e[1:] /= spec1_ej[1:, 0]
spec2_ej = np.loadtxt('%s_lr_spec.dat' % name)
E = lr.lr_transitions.get_transitions()[0][0]
data2_e = spec2_ej[:, 5] / E

# One can decrease the tolerance by decreasing the time step
# and other parameters
equal(data1_e, data2_e, 0.01)
