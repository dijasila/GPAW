import numpy as np
import pickle as pckl
from ase import Atoms
from gpaw import GPAW
from gpaw.wavefunctions.pw import PW
from ase.dft.kpoints import monkhorst_pack
from gpaw.response.df2 import DielectricFunction
from ase.parallel import paropen as open
from gpaw.test import equal, findpeak

# Comparing the plasmon peaks found in bulk sodium for two different
# atomic structures. Testing for idential plasmon peaks. Not using 
# physical sodium cell. 

a_v = 4.23/2.0
a1 = Atoms('Na',
           scaled_positions=[[0, 0, 0]],
           cell=(a_v, a_v, a_v),
           pbc=True)

# Expanding along x-direction
a2 = Atoms('Na2',
           scaled_positions=[[0, 0, 0], [0.5, 0, 0]],
           cell=(2*a_v, a_v, a_v),
           pbc=True)

# Kpoint sampling should be halved in the expanded direction. 
k1 = monkhorst_pack((8, 8, 8)) + np.array([1./16, 1./16, 1./16])
k2 = monkhorst_pack((4, 8, 8)) + np.array([1./8, 1./16, 1./16])

a1.calc = GPAW(gpts=(10, 10, 10),
               basis='dzp',
               mode=PW(300),
               kpts=k1,
               parallel={'band': 1},
               )

a2.calc = GPAW(gpts=(20, 10, 10),
               basis='dzp',
               mode=PW(300),
               kpts=k2,
               parallel={'band': 1},
               ) 

a1.get_potential_energy()  
a2.get_potential_energy()

# Use twice as many bands for expanded structure
a1.calc.diagonalize_full_hamiltonian(nbands=20)
a2.calc.diagonalize_full_hamiltonian(nbands=40)

a1.calc.write('gs_Na_small.gpw', 'all')
a2.calc.write('gs_Na_large.gpw', 'all')

# Calculate the dielectric functions
df1 = DielectricFunction(calc='gs_Na_small.gpw',
                         omegamax=15,
                         domega0=0.05,
                         hilbert=True,
                         ecut=150)

df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')

df2 = DielectricFunction(calc='gs_Na_large.gpw',
                         omegamax=15,
                         domega0=0.05,
                         hilbert=True,
                         ecut=150)

df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')

# Compare plasmon frequencies and intensities
w_w = df1.chi0.omega_w
w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
equal(w1, w2, 1e-2)
equal(I1, I2, 1e-3)

w1, I1 = findpeak(w_w, -(1. / df1LFCy).imag)
w2, I2 = findpeak(w_w, -(1. / df2LFCy).imag)
equal(w1, w2, 1e-2)
equal(I1, I2, 1e-3)

w1, I1 = findpeak(w_w, -(1. / df1LFCz).imag)
w2, I2 = findpeak(w_w, -(1. / df2LFCz).imag)
equal(w1, w2, 1e-2)
equal(I1, I2, 1e-3)
