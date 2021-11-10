from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from ase.build import molecule
from ase import Atoms
from sys import argv
from os.path import exists

from ase.collections import g2

#atoms = molecule('H2')
atoms = Atoms('H2', positions = ((0,0,0),(0.3,0.8,1.0)))
atoms.center(vacuum=5.5)

#algs = [ 'RIVFullBasisDebug', 'RI-MPV' ]
algs = [ 'RI-MPV' ]

for alg in algs:
    calc = GPAW(h=0.2,
                poissonsolver=PoissonSolver(remove_moment=1+3+5),
                mode='lcao', xc='EXX:backend=aux-lcao:algorithm=%s' % alg, basis='dzp')
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
