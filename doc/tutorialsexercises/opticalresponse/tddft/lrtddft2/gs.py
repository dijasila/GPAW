# web-page: gs.out
# Start
from ase.io import read
from gpaw import GPAW

atoms = read('r-methyloxirane.xyz')
atoms.center(vacuum=8)

calc = GPAW(h=0.2,
            nbands=14,
            xc='LDA',
            poissonsolver={'MomentCorrectionPoissonSolver':
                           {'poissonsolver': 'fast',
                            'moment_corrections': 1 + 3 + 5}
                           },
            convergence={'bands': 'occupied'},
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
