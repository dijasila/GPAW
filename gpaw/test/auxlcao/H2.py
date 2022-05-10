from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from ase.build import molecule, bulk
from ase import Atoms
from sys import argv
from os.path import exists
from ase.collections import g2
import numpy as np
from gpaw.test import equal

for R in [ 0.7 ]:
    atoms = molecule('H2')
    alg = 'RI-MPV'
    atoms.center(vacuum=2)
    calc = GPAW(h=0.20,
                mode='lcao', 
                xc='PBE0:backend=aux-lcao:algorithm=%s' % alg, 
                basis='dzp', txt='None')
    atoms.calc = calc
    atoms.get_potential_energy()
    equal([-11.99158365,   3.27369088], calc.get_eigenvalues(), 1e-6)
    equal(-6.554025635093129, calc.get_potential_energy(), 1e-5)

