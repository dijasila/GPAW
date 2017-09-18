from __future__ import print_function
from ase.build import molecule

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.overlap import Overlap

h = 0.4
box = 2
nbands = 2
txt = '-'
txt = None

H2 = Cluster(molecule('H2'))
H2.minimal_box(box, h)

c1 = GPAW(h=h, txt=None, nbands=nbands)
c1.calculate(H2)

H2[1].position[1] += 0.05
c2 = GPAW(h=h, txt=None, nbands=nbands + 1)
c2.calculate(H2)

print(Overlap(c1).pseudo(c2))

