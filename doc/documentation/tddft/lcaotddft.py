# Simplest example of use of LCAO-TDDFT code

from ase import Atoms
from gpaw import GPAW
from ase.optimize import BFGS 
from gpaw.tddft import photoabsorption_spectrum
from gpaw import PoissonSolver

# Sodium dimer
atoms = Atoms('Na2', positions=[[0.0,0.0,0.0],[3.0,0.0,0.0]])
atoms.center(vacuum=5.0)

from gpaw.lcaotddft import LCAOTDDFT

# Increase accuragy of density for ground state
convergence = {'density':1e-7} 

# Increase accuracy of Poisson Solver and apply multipole corrections up to l=2
poissonsolver = PoissonSolver(eps=1e-20, remove_moment=1+3+5)

td_calc = LCAOTDDFT(setups={'Na':'1'}, basis='1.dzp', xc='LDA', h=0.3, nbands=1,
                    convergence=convergence, poissonsolver=poissonsolver)

atoms.set_calculator(td_calc)
# Relax the ground state
atoms.get_potential_energy()

td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(10, 500, 'Na2.dm')

photoabsorption_spectrum('Na2.dm','Na2.spec', width=0.4)

