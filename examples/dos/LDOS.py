#!/usr/bin/env python
from gpaw import Calculator
from gpaw.utilities.dos import DOS
from ase import *
import numpy as num


########################################
#This script produces the projected density of states
########################################
filename='Fe_nonmag.gpw'
calc = Calculator(filename)
##############################
#Set up the calculator
##############################

ldos = LDOS(calc)
energies = ldos.get_energies()
dos_ie = ldos.get_l_d_o_s(a=1)

# Plot LDOS
import pylab
plot(energies, dos_ie[0])
show()


