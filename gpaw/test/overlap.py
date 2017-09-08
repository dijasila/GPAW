from __future__ import print_function
import numpy as np
from ase.build import molecule

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.overlap import Overlap
from gpaw.lrtddft import LrTDDFT

"""Evaluate the overlap between two independent calculations

Differences are forced by different eigensolvers and differing number
of Kohn-Sham states.
""" 

h = 0.4
box = 2
nbands = 4
txt = '-'
txt = None
np.set_printoptions(precision=3, suppress=True)

H2 = Cluster(molecule('H2'))
H2.minimal_box(box, h)

c1 = GPAW(h=h, txt=txt, eigensolver='dav', nbands=nbands,
          convergence={'eigenstates':nbands})
c1.calculate(H2)
lr1 = LrTDDFT(c1)

def show(c2):
    c2.calculate(H2)
    lr2 = LrTDDFT(c2)
    ov = Overlap(c1).pseudo(c2)
    print('wave function overlap:\n', ov)
    ovkss = lr1.kss.overlap(ov, lr2.kss)
    print('KSSingles overlap:\n', ovkss)
    ovlr = lr1.overlap(ov, lr2)
    print('LrTDDFT overlap:\n', ovlr)


print('cg --------')
c2 = GPAW(h=h, txt=txt, eigensolver='cg', nbands=nbands + 1,
          convergence={'eigenstates':nbands + 1})
show(c2)

print('spin --------')
c2 = GPAW(h=h, txt=txt, spinpol=True, nbands=nbands + 1,
          convergence={'eigenstates':nbands + 1})
try:
    show(c2)
    raise
except AssertionError:
    print('Not ready')
