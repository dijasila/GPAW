"""
Calculate the magnetic response in iron using ALDA.
The kernel is represented on a grid in real-space.
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
q_c = [0.0, 0.0, 0.0]  # Gamma point
frq_w = np.linspace(0.000, 0.200, 26)
Kxc = 'ALDA'
ecut = 300
eta = 0.01

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

chiM0_w, chiM_w = tms.get_dynamic_susceptibility(q_c=q_c, xc=Kxc,
                                                 rshe=None,
                                                 filename='iron_dsus'
                                                 + '_G.csv')

t3 = time.time()

parprint('Ground state calculation took', (t2 - t1) / 60, 'minutes')
parprint('Excited state calculation took', (t3 - t2) / 60, 'minutes')

world.barrier()

# Part 3: identify magnon peak in scattering function
d = np.loadtxt('iron_dsus_G.csv', delimiter=', ')

wpeak, Ipeak = findpeak(d[:, 0], - d[:, 4])

mw = (wpeak + d[0, 0]) * 1000

# Part 4: compare new results to test values
test_mw = 69  # meV
test_Ipeak = 67  # a.u.

# Magnon peak:
equal(mw, test_mw, eta * 150)

# Scattering function intensity:
equal(Ipeak, test_Ipeak, 1.5)
