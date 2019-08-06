"""
Calculate the magnetic response in iron using ALDA.
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
a = 2.867
mm = 2.21

# Part 2: magnetic response calculation
q_c = [0.0, 0.0, 0.0]  # Gamma-point
frq_w = np.linspace(0.150, 0.350, 26)
Kxc = 'ALDA'
ecut = 300
eta = 0.01
rshecc_c = [0.99, 0.999]  # test different levels of expansion

# ------------------- Script ------------------- #

# Part 1: ground state calculation

t1 = time.time()

Febcc = bulk('Fe', 'bcc', a=a)
Febcc.set_initial_magnetic_moments([mm])

calc = GPAW(xc=xc,
            mode=PW(pw),
            kpts=monkhorst_pack((kpts, kpts, kpts)),
            nbands=nb,
            idiotproof=False,
            parallel={'band': 1})

Febcc.set_calculator(calc)
Febcc.get_potential_energy()
calc.write('Fe', 'all')
t2 = time.time()

# Part 2: magnetic response calculation
tms = TransverseMagneticSusceptibility(calc='Fe',
                                       frequencies=frq_w,
                                       eta=eta,
                                       ecut=ecut)

for c, rshecc in enumerate(rshecc_c):
    chiM0_w, chiM_w = tms.get_dynamic_susceptibility(q_c=q_c, xc=Kxc,
                                                     rshe=rshecc,
                                                     filename='iron_dsus'
                                                     + '_G%d.csv' % (c + 1))

t3 = time.time()

parprint('Ground state calculation took', (t2 - t1) / 60, 'minutes')
parprint('Excited state calculation took', (t3 - t2) / 60, 'minutes')

world.barrier()

# Part 3: identify magnon peak in scattering functions
d1 = np.loadtxt('iron_dsus_G1.csv', delimiter=', ')
d2 = np.loadtxt('iron_dsus_G2.csv', delimiter=', ')

wpeak1, Ipeak1 = findpeak(d1[:, 0], - d1[:, 4])
wpeak2, Ipeak2 = findpeak(d2[:, 0], - d2[:, 4])

mw1 = (wpeak1 + d1[0, 0]) * 1000
mw2 = (wpeak2 + d2[0, 0]) * 1000

# Part 4: compare new results to test values
test_mw1 = 242  # meV
test_mw2 = 244  # meV
test_Ipeak1 = 66  # a.u.
test_Ipeak2 = 69  # a.u.

# Magnon peak:
equal(mw1, test_mw1, eta * 200)
equal(mw2, test_mw2, eta * 200)

# Scattering function intensity:
equal(Ipeak1, test_Ipeak1, 3.5)
equal(Ipeak2, test_Ipeak2, 3.5)
