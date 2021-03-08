from gpaw import *
from ase.io import *

from gpaw import PoissonSolver
poissonsolver = PoissonSolver(eps=1e-20, remove_moment=1 + 3 + 5)


atoms = read('methyl-oxirane.xyz')
#atoms.set_cell((L, L, L))
atoms.center(vacuum=6.0)

calc =  GPAW(mode='lcao',xc='PBE',nbands=82, h=0.2, basis='dzp',
            poissonsolver=poissonsolver,
            convergence={'density': 1e-12})
atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.write('methyl-oxirane.gpw', mode='all')

