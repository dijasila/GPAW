from __future__ import print_function
from ase.build import molecule

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.overlap import Overlap
from gpaw.lrtddft import LrTDDFT

h = 0.4
box = 2
nbands = 4
txt = '-'
txt = None

H2 = Cluster(molecule('H2'))
H2.minimal_box(box, h)

c1 = GPAW(h=h, txt=txt, nbands=nbands, convergence={'eigenstates':nbands})
c1.calculate(H2)
lr1 = LrTDDFT(c1)

c2 = GPAW(h=h, txt=txt, eigensolver='cg', nbands=nbands,
          convergence={'eigenstates':nbands})
c2.calculate(H2)
lr2 = LrTDDFT(c2)

ov = Overlap(c1).pseudo(c2)
print('wave function overlap:\n', ov)
ovkss = lr1.kss.overlap(ov, lr2.kss)
print('KSSingles overlap:\n', ovkss)
ovlr = lr1.overlap(ov, lr2)
print('LrTDDFT overlap:\n', ovlr)
