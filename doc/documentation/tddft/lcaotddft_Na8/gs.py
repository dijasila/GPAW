from ase import Atoms
from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver

# Sodium atom chain
atoms = Atoms('Na8',
              positions=[[i * 3.0, 0, 0] for i in range(8)],
              cell=[33.6, 12.0, 12.0])
atoms.center()

# Use an advanced Poisson solver
ps = ExtraVacuumPoissonSolver(gpts=(512, 256, 256),
                              poissonsolver_large=PoissonSolver(),
                              coarses=2,
                              poissonsolver_small=PoissonSolver())

# Ground-state calculation
calc = GPAW(mode='lcao', h=0.3, basis='pvalence.dz', xc='LDA', nbands=6,
            setups={'Na': '1'},
            poissonsolver=ps,
            convergence={'density': 1e-12},
            txt='gs.out')
atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
