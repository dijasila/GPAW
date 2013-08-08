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

a = 4.043
atoms = bulk('Al', 'fcc', a=a)
atoms.center()
calc = GPAW(h=0.2,
            kpts=(4,4,4),
            mode='lcao',
            basis='dzp',
            xc='LDA')

atoms.set_calculator(calc)
t1 = time.time()
atoms.get_potential_energy()
t2 = time.time()
calc.write('Al.gpw','all')

t3 = time.time()

# Excited state calculation
q = np.array([1/4.,0.,0.])
w = np.linspace(0, 24, 241)
    
df = DF(calc='Al.gpw', q=q, w=w, eta=0.2, ecut=50)
#df.write('Al.pckl')
df.get_EELS_spectrum(filename='EELS_Al_lcao')
df.check_sum_rule()
    
t4 = time.time()

print 'For ground  state calc, it took', (t2 - t1) / 60, 'minutes'
print 'For writing gpw, it took', (t3 - t2) / 60, 'minutes'
print 'For excited state calc, it took', (t4 - t3) / 60, 'minutes'

d = np.loadtxt('EELS_Al_lcao')
wpeak = 15.9 # eV
x, y = findpeak(d[:, 1], 0.1)
print(x, y)
equal(x, wpeak, 0.05)
equal(y, 28.3, 0.15)
x, y = findpeak(d[:, 2], 0.1)
print(x, y)
equal(x, wpeak, 0.05)
equal(y, 26.6, 0.15)
