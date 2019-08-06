"""
Calculate the magnetic response in iron using ALDA.

Tests whether the magnon energies and scattering intensities
have changed for:
 * Different kernel calculation strategies
 * Different chi0 transitions summation strategies
"""

# Workflow modules
import numpy as np

# Script modules
import time

from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.test import findpeak, equal
from gpaw.mpi import world

# ------------------- Inputs ------------------- #

# Part 1: ground state calculation
xc = 'LDA'
kpts = 4
nb = 6
pw = 300
conv = {'density': 1.e-8,
        'forces': 1.e-8}
a = 2.867
mm = 2.21

# Part 2: magnetic response calculation
q_c = [0.0, 0.0, 1 / 4.]
fxc = 'ALDA'
ecut = 300
eta = 0.01

# Test different kernel and summation strategies
strat_sd = [(None, 'pairwise', False),  # rshe, bandsummation, memory_safe
            (0.99, 'pairwise', False),
            (0.99, 'pairwise', True),
            (0.999, 'pairwise', False),
            (0.999, 'double', False)]
frq_sw = [np.linspace(0.160, 0.320, 21),
          np.linspace(0.320, 0.480, 21),
          np.linspace(0.320, 0.480, 21),
          np.linspace(0.320, 0.480, 21),
          np.linspace(0.320, 0.480, 21)]

# ------------------- Script ------------------- #

# Part 1: ground state calculation

t1 = time.time()

Febcc = bulk('Fe', 'bcc', a=a)
Febcc.set_initial_magnetic_moments([mm])

calc = GPAW(xc=xc,
            mode=PW(pw),
            kpts=monkhorst_pack((kpts, kpts, kpts)),
            nbands=nb,
            convergence=conv,
            symmetry={'point_group': False},
            idiotproof=False,
            parallel={'band': 1})

Febcc.set_calculator(calc)
Febcc.get_potential_energy()
# calc.write('Fe', 'all')  # remove XXX
t2 = time.time()

# Part 2: magnetic response calculation

for s, ((rshe, bandsummation, memory_safe), frq_w) in enumerate(zip(strat_sd,
                                                                    frq_sw)):
    tms = TransverseMagneticSusceptibility(calc,  # 'Fe',  # remove me XXX
                                           fxc=fxc,
                                           eta=eta,
                                           ecut=ecut,
                                           bandsummation=bandsummation,
                                           fxckwargs={'rshe': rshe},
                                           memory_safe=memory_safe,
                                           nblocks=2)
    tms.get_macroscopic_component('+-', q_c, frq_w,
                                  filename='iron_dsus' + '_G%d.csv' % (s + 1))
    tms.write_timer()

t3 = time.time()

parprint('Ground state calculation took', (t2 - t1) / 60, 'minutes')
parprint('Excited state calculations took', (t3 - t2) / 60, 'minutes')

world.barrier()

# Part 3: identify magnon peak in scattering functions
d1 = np.loadtxt('iron_dsus_G1.csv', delimiter=', ')
d2 = np.loadtxt('iron_dsus_G2.csv', delimiter=', ')
d3 = np.loadtxt('iron_dsus_G3.csv', delimiter=', ')
d4 = np.loadtxt('iron_dsus_G4.csv', delimiter=', ')
d5 = np.loadtxt('iron_dsus_G5.csv', delimiter=', ')

wpeak1, Ipeak1 = findpeak(d1[:, 0], d1[:, 4])
wpeak2, Ipeak2 = findpeak(d2[:, 0], d2[:, 4])
wpeak3, Ipeak3 = findpeak(d3[:, 0], d3[:, 4])
wpeak4, Ipeak4 = findpeak(d4[:, 0], d4[:, 4])
wpeak5, Ipeak5 = findpeak(d5[:, 0], d5[:, 4])

mw1 = (wpeak1 + d1[0, 0]) * 1000
mw2 = (wpeak2 + d2[0, 0]) * 1000
mw3 = (wpeak3 + d3[0, 0]) * 1000
mw4 = (wpeak4 + d4[0, 0]) * 1000
mw5 = (wpeak5 + d5[0, 0]) * 1000

# Part 4: compare new results to test values
test_mw1 = 245.59  # meV
test_mw2 = 401.01  # meV
test_mw4 = 402.38  # meV
test_Ipeak1 = 57.56  # a.u.
test_Ipeak2 = 58.46  # a.u.
test_Ipeak4 = 56.15  # a.u.

# Different kernel strategies should remain the same
# Magnon peak:
equal(mw1, test_mw1, eta * 100)
equal(mw2, test_mw2, eta * 100)
equal(mw4, test_mw4, eta * 100)

# Scattering function intensity:
equal(Ipeak1, test_Ipeak1, 1.5)
equal(Ipeak2, test_Ipeak2, 1.5)
equal(Ipeak4, test_Ipeak4, 1.5)

# The vectorized and un-vectorized integration methods should give the same
equal(mw2, mw3, eta * 100)
equal(Ipeak2, Ipeak3, 1.5)

# The two transitions summation strategies should give identical results
equal(mw4, mw5, eta * 100)
equal(Ipeak4, Ipeak5, 1.5)
