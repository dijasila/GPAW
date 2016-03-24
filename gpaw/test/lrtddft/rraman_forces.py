import sys
from ase import Atoms, Atom
from ase.vibrations.resonant_raman import ResonantRaman

from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.mpi import world

txt = '-'
txt = None
load = True
load = False
xc = 'LDA'

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))

gsname = exname = 'rraman'
rr = ResonantRaman(H2, KSSingles, gsname=gsname, exname=exname, 
                   verbose=True,)
rr.summary(omega=5, method='frederiksen')
