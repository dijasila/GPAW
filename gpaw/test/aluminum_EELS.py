import numpy as np
import sys
import os
import time
from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull
from gpaw.test import findpeak, equal


if rank != 0:
  sys.stdout = devnull 

assert size <= 4**3

# Ground state calculation

t1 = time.time()

a = 4.043
atoms = bulk('Al', 'fcc', a=a)
atoms.center()
calc = GPAW(h=0.2,
            kpts=(4,4,4),
            parallel={'domain':1,
                      'band':1},
            idiotproof=False,  # allow uneven distribution of k-points
            xc='LDA')

atoms.set_calculator(calc)
atoms.get_potential_energy()
t2 = time.time()

# Excited state calculation
q = np.array([1/4.,0.,0.])
w = np.linspace(0, 24, 241)

df = DF(calc=calc, q=q, w=w, eta=0.2, ecut=(50,50,50))
df.get_EELS_spectrum(filename='EELS_Al')
df.check_sum_rule()
df.write('Al.pckl')

t3 = time.time()

print 'For ground  state calc, it took', (t2 - t1) / 60, 'minutes'
print 'For excited state calc, it took', (t3 - t2) / 60, 'minutes'

d = np.loadtxt('EELS_Al')
wpeak = 15.7 # eV
x, y = findpeak(d[:, 1], 0.1)
print(x, y)
equal(x, wpeak, 0.05)
equal(y, 29.0, 0.15)
x, y = findpeak(d[:, 2], 0.1)
print(x, y)
equal(x, wpeak, 0.05)
equal(y, 26.6, 0.15)
