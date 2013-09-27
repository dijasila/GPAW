from ase import Atoms
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal


txt = None
txt = '-'

s = Atoms('H', magmoms=[1])
s.center(vacuum=4.0)

c = GPAW(xc='TPSS',
         mode='pw',
         txt=txt)
c.calculate(s)

cpbe = GPAW(xc='PBE', mode='pw', txt=txt)
cpbe.calculate(s)
cpbe.set(xc='TPSS')
cpbe.calculate()

print "Energy difference", (cpbe.get_potential_energy() - 
                            c.get_potential_energy())
equal(cpbe.get_potential_energy(), c.get_potential_energy(), 0.002)
