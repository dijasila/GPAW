import numpy as np
import sys
import time

from ase.units import Bohr
from ase.lattice import bulk
from gpaw import GPAW, FermiDirac, PW
from gpaw.atom.basis import BasisMaker
from gpaw.response.df2 import DielectricFunction
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

# GS Calculation One
a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode='pw',
            kpts=(4,4,4),
            occupations=FermiDirac(0.001))

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C.gpw','all')

# Macroscopic dielectric constant calculation
#q = np.array([0.0, 0.00001, 0.])  It'not needed anymore in the new version of the code
w = np.linspace(0, 24., 241)

df = DielectricFunction(calc='C.gpw', omega_w=(0.,), eta=0.001,
        ecut=50, txt='diamond_df_out_new.txt')
eM1, eM2 = df.get_macroscopic_dielectric_constant(direction='x')

print eM1, eM2

eM1_ = 5.83540485038
eM2_ = 5.73382717669

if (np.abs(eM1 - eM1_) > 1e-5 or
    np.abs(eM2 - eM2_) > 1e-5):
    print eM1, eM2
    raise ValueError('Macroscopic dielectric constant not correct ! ')


# Absorption spectrum calculation
del df
df = DielectricFunction(calc='C.gpw', omega_w=w, eta=0.25,
        ecut=50, txt='C_df.out')
df.get_absorption_spectrum()
df.check_sum_rule()
df.write('C_df.pckl')
