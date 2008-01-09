"""Bulk Al(fcc) test"""

from ase import *
from gpaw import *

name = 'Al-fcc'
a = 4.05   # fcc lattice paramter
b = a / 2 
bulk = Atoms(symbols='4Al',
             positions=[(0, 0, 0),
                        (b, b, 0),
                        (0, b, b),
                        (b, 0, b)],
             cell=(a, a, a),
             pbc=True)

k = 4
calc = Calculator(nbands=16, txt=name + '-k.txt')
bulk.set_calculator(calc)

# Make a plot of the convergence with respect to k-points
calc.set(h=0.3)
f = open(name + '-k.dat', 'w')
#for k in [2, 4, 6, 8, 10, 12]: 
for k in [2, 4, 6]:
    calc.set(kpts=(k, k, k))
    energy = bulk.get_potential_energy() 
    print k, energy
    print >> f, k, energy

# Make a plot of the convergence with respect to grid spacing
k = 4
calc.set(kpts=(k, k, k), txt=name + '-h.txt')
f = open(name + '-h.dat', 'w')
#for h in [0.5, 0.3, 0.25, 0.2, 0.15]:
for g in [12, 16]:
    h = a / g
    calc.set(h=h)
    energy = bulk.get_potential_energy() 
    print h, energy
    print >> f, h, energy
