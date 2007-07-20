#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import gpaw

a = 5.0
H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=1)],
                periodic=False,
                cell=(a, a, a))

H.SetCalculator(gpaw(nbands=1, h=0.2))
e = H.GetPotentialEnergy()
