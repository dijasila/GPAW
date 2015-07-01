import numpy as np
from ase import *
from ase.dft.kpoints import monkhorst_pack
from ase.lattice.hexagonal import Graphite
from gpaw import *
from gpaw.wavefunctions.pw import PW

a = 2.46 # Lattice parameter of graphene
d = 3.34

calc = GPAW(xc='PBE',
            mode=PW(600),
            kpts=(10,10,6),
            occupations=FermiDirac(width=0.01),
            txt='graph.txt')

bulk = Graphite(symbol='C', latticeconstant={'a':a, 'c':2*d}, pbc=True)
bulk.set_calculator(calc)
bulk.get_potential_energy()

calc.set(kpts={'size': (4,4,4), 'gamma': True},
         nbands=20,
         convergence={'bands': 16},
         symmetry='off',
         fixdensity=True)
calc.get_potential_energy()
calc.write('graph.gpw', mode='all')
