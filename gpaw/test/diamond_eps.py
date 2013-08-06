import numpy as np
import sys
import time

from ase.units import Bohr
from ase.structure import bulk
from gpaw import GPAW, FermiDirac
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull
from gpaw.test import findpeak, equal


if rank != 0:
  sys.stdout = devnull 

# GS Calculation One
a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a)

nbands = 8

calc = GPAW(h=0.2,
            kpts=(2,2,2),
            nbands = nbands+5,
            eigensolver='cg',
            occupations=FermiDirac(0.001),
            convergence={'bands':nbands})

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C2.gpw','all')

# Macroscopic dielectric constant calculation
q = np.array([0.0, 0.00001, 0.])
w = np.linspace(0,15,150)

df = DF(calc='C2.gpw',
        q=q,
        w=(0.,),
        eta=0.001,
        nbands=nbands,
        ecut=50,
        hilbert_trans=False,
        optical_limit=True)
eM1, eM2 = df.get_macroscopic_dielectric_constant(xc='ALDA')

if np.abs(eM2 - 7.91) > 1e-1:
    raise ValueError("Incorrect value for Diamond dielectric constant with ALDA Kernel %.4f" % (eM2))

# Dielectric function
df = DF(calc='C2.gpw',
        q=q,
        w=w,
        eta=0.40,
        nbands=nbands,
        xc='Bootstrap',
        ecut=100,
        hilbert_trans=False,
        optical_limit=True)
df.get_absorption_spectrum(filename='C2.dat')

spect = np.loadtxt('C2.dat.y')
eps2_max = findpeak(spect[:, 6], dx=spect[1, 0])[0]
equal(eps2_max, 9.19, 0.05,
      msg='Incorrect position for Diamond dielectric function peak with ' +
      'Bootstrap Kernel')

# RPA:
# With kpts=(12,12,12) and bands=64, ecut=250eV, this script gives 5.56
# Value from PRB 73, 045112 with kpts=(12,12,12) and bands=64: 5.55
# ALDA:
# With kpts=(12,12,12) and bands=64, ecut=250eV, this script gives 5.82 
# Value from PRB 73, 045112 with kpts=(12,12,12) and bands=64: 5.82
