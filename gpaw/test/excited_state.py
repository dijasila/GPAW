import os
import sys
from ase import Atom, Atoms
from ase.units import Bohr
from ase.parallel import parprint
from gpaw import GPAW
from gpaw.test import equal
from gpaw.lrtddft import LrTDDFT
from gpaw.mpi import world 
from gpaw.lrtddft.excited_state import ExcitedState

txt='-'
txt='/dev/null'

R=0.7 # approx. experimental bond length
a = 3.0
c = 4.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))
calc = GPAW(xc='PBE', nbands=3, spinpol=False, txt=txt)
H2.set_calculator(calc)

xc='LDA'
lr = LrTDDFT(calc, xc=xc)

# excited state with forces
accuracy = 0.01
exst = ExcitedState(lr, 0, d=0.01,
        parallel=2,
        txt=sys.stdout,
      )

forces = exst.get_forces(H2)
for c in range(2):
    equal(forces[0,c], 0.0, accuracy)
    equal(forces[1,c], 0.0, accuracy)
equal(forces[0, 2] + forces[1, 2], 0.0, accuracy)
