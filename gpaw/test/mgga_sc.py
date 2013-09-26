from ase import Atom
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal

h=0.2
txt = None
txt = '-'

s = Cluster([Atom('H')])
s.minimal_box(4., h=h)
s.set_initial_magnetic_moments([1])

c = GPAW(xc='TPSS', h=h, #nbands=5,
         txt=txt, 
         #eigensolver='cg', 
         maxiter=300)
c.calculate(s)

cpbe = GPAW(xc='PBE', h=h, nbands=5, txt=txt)
cpbe.calculate(s)
cpbe.set(xc='TPSS')
cpbe.calculate()

print "Energy difference", (cpbe.get_potential_energy() - 
                            c.get_potential_energy())
equal(cpbe.get_potential_energy(), c.get_potential_energy(), 0.002)
