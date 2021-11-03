from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from ase.build import molecule
from ase import Atoms
from sys import argv
from os.path import exists

from ase.collections import g2

atoms = Atoms('H2', positions=[[0,0,0],[0.7,0,0]])
atoms.center(vacuum=5)
calc = GPAW(h=0.16,
            poissonsolver=PoissonSolver(remove_moment=1+3+5),
            mode='lcao', xc='PBE0:backend=aux-lcao', basis='dzp')
atoms.set_calculator(calc)
atoms.get_potential_energy()
