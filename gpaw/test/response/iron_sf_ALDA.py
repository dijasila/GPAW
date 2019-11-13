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
# rshe, bandsummation, bundle_integrals, bundle_kptpairs
strat_sd = [(None, 'pairwise', True, True),
            (0.99, 'pairwise', True, True),
            (0.99, 'pairwise', False, True),
            (0.99, 'pairwise', True, False),
            (0.999, 'pairwise', True, True),
            (0.999, 'double', True, True)]
frq_sw = [np.linspace(0.160, 0.320, 21),
          np.linspace(0.320, 0.480, 21),
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
            parallel={'domain': 1})

Febcc.set_calculator(calc)
Febcc.get_potential_energy()
t2 = time.time()

# Part 2: magnetic response calculation

for s, ((rshe, bandsummation, bundle_integrals, bundle_kptpairs),
        frq_w) in enumerate(zip(strat_sd, frq_sw)):
    tms = TransverseMagneticSusceptibility(calc,
                                           fxc=fxc,
                                           eta=eta,
                                           ecut=ecut,
                                           bandsummation=bandsummation,
                                           fxckwargs={'rshe': rshe},
                                           bundle_integrals=bundle_integrals,
                                           bundle_kptpairs=bundle_kptpairs,
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
d6 = np.loadtxt('iron_dsus_G6.csv', delimiter=', ')

wpeak1, Ipeak1 = findpeak(d1[:, 0], d1[:, 4])
wpeak2, Ipeak2 = findpeak(d2[:, 0], d2[:, 4])
wpeak3, Ipeak3 = findpeak(d3[:, 0], d3[:, 4])
wpeak4, Ipeak4 = findpeak(d4[:, 0], d4[:, 4])
wpeak5, Ipeak5 = findpeak(d5[:, 0], d5[:, 4])
wpeak6, Ipeak6 = findpeak(d6[:, 0], d6[:, 4])

mw1 = (wpeak1 + d1[0, 0]) * 1000
mw2 = (wpeak2 + d2[0, 0]) * 1000
mw3 = (wpeak3 + d3[0, 0]) * 1000
mw4 = (wpeak4 + d4[0, 0]) * 1000
mw5 = (wpeak5 + d5[0, 0]) * 1000
mw6 = (wpeak6 + d6[0, 0]) * 1000

# Part 4: compare new results to test values
test_mw1 = 234.63  # meV
test_mw2 = 397.33  # meV
test_mw5 = 398.83  # meV
test_Ipeak1 = 56.74  # a.u.
test_Ipeak2 = 55.80  # a.u.
test_Ipeak5 = 58.23  # a.u.

# Different kernel strategies should remain the same
# Magnon peak:
equal(mw1, test_mw1, eta * 250)
equal(mw2, test_mw2, eta * 250)
equal(mw5, test_mw5, eta * 250)

# Scattering function intensity:
equal(Ipeak1, test_Ipeak1, 2.5)
equal(Ipeak2, test_Ipeak2, 2.5)
equal(Ipeak5, test_Ipeak5, 2.5)

# The bundled and unbundled integration methods should give the same
equal(mw2, mw3, eta * 100)
equal(Ipeak2, Ipeak3, 1.0)

# The bundled and unbundled data extraction should give the same result
equal(mw2, mw4, eta * 100)
equal(Ipeak2, Ipeak4, 1.0)

# The two transitions summation strategies should give identical results
equal(mw5, mw6, eta * 100)
equal(Ipeak5, Ipeak6, 1.0)
