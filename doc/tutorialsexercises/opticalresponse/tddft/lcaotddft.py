# Simple example of the use of LCAO-TDDFT code

# Sodium dimer
from ase.build import molecule
atoms = molecule('Na2')
atoms.center(vacuum=6.0)

# Poisson solver with multipole corrections up to l=2
from gpaw import PoissonSolver
from gpaw.poisson_moment import MomentCorrectionPoissonSolver
poissonsolver = MomentCorrectionPoissonSolver(poissonsolver=PoissonSolver(),
                                              moment_corrections=1 + 3 + 5)

# Ground-state calculation
from gpaw import GPAW
calc = GPAW(mode='lcao', h=0.3, basis='dzp',
            setups={'Na': '1'},
            poissonsolver=poissonsolver,
            convergence={'density': 1e-12})
atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
# Read converged ground-state file
td_calc = LCAOTDDFT('gs.gpw')
# Attach any data recording or analysis tools
DipoleMomentWriter(td_calc, 'dm.dat')
# Kick
td_calc.absorption_kick([0.0, 0.0, 1e-5])
# Propagate
td_calc.propagate(10, 3000)
# Save the state for restarting later
td_calc.write('td.gpw', mode='all')

# Analyze the results
from gpaw.tddft.spectrum import photoabsorption_spectrum
photoabsorption_spectrum('dm.dat', 'spec.dat')
