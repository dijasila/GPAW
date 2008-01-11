#!/usr/bin/env python
from gpaw import Calculator
from ase import *
import numpy as num


########################################
#This script produces the projected density of states
########################################
filename='Fe_nonmag.gpw'
calc = Calculator(filename)

energies, ldos = calc.GetOrbitalLDOS(a=1, spin=0, angular='s')

# Plot LDOS
import pylab
plot(energies, ldos)
show()


