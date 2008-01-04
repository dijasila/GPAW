"""Bulk Al(fcc) test"""

from ase import *
from gpaw import *

filename = 'Al-fcc'

a = 4.05   # fcc lattice paramter
b = a / 2 

bulk = Atoms([Atom('Al', (0, 0, 0)),
              Atom('Al', (b, b, 0)),
              Atom('Al', (0, b, b)),
              Atom('Al', (b, 0, b)),],
             cell=(a, a, a),
             pbc=(1, 1, 1))

k = 4
calc = Calculator(nbands=16,              # number of electronic bands
                  h=0.2,                  # grid spacing
                  kpts=(k, k, k),         # k-points
                  txt=filename + '.txt')  # output file
bulk.set_calculator(calc)

energy = bulk.get_potential_energy()
calc.write(filename+'.gpw')
print energy
