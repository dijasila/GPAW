from ase import Atoms
from gpaw import GPAW, PW
from gpaw.response.df import DielectricFunction
from gpaw.test import equal, findpeak

# Comparing the plasmon peaks found in bulk sodium for two different
# atomic structures. Testing for idential plasmon peaks. Not using
# physical sodium cell.

a = 4.23 / 2.0
a1 = Atoms('Na',
           scaled_positions=[[0, 0, 0]],
           cell=(a, a, a),
           pbc=True)

a1.calc = GPAW(gpts=(10, 10, 10),
               mode=PW(300),
               kpts={'size': (10, 10, 10), 'gamma': True},
               parallel={'band': 1},
               txt='small.txt')

a1.get_potential_energy()

# Use twice as many bands for expanded structure
a1.calc.diagonalize_full_hamiltonian(nbands=20)

a1.calc.write('gs_Na.gpw', 'all')

# Calculate the dielectric functions
df1 = DielectricFunction('gs_Na.gpw',
                         hilbert=True,
                         nblocks=1,
                         ecut=150)

df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')

df2 = DielectricFunction('gs_Na.gpw',
                         hilbert=True,
                         nblocks=4,
                         ecut=150)

df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')

df3 = DielectricFunction('gs_Na.gpw',
                         hilbert=True,
                         nblocks=8,
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
