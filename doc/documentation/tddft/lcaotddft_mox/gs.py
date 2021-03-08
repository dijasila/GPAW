from ase.io import read
from gpaw import GPAW, PoissonSolver

atoms = read('methyloxirane.xyz')
atoms.center(vacuum=6.0)

poissonsolver = PoissonSolver(remove_moment=1 + 3 + 5)

calc = GPAW(mode='lcao',
            xc='PBE',
            nbands=16,
            h=0.3,
            basis='dzp',
            poissonsolver=poissonsolver,
            convergence={'density': 1e-12},
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
