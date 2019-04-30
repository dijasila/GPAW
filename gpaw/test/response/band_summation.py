"""Compare results with and without spin-conserving time-reversal symmetry"""

# Workflow modules
import numpy as np

# Script modules
import time

from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.response.df import DielectricFunction
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.test import findpeak, equal
from gpaw.mpi import world

# --------------- Inputs --------------- #

# Part 1: aluminum
# Part 1.1: aluminum ground state calculation
aluxc = 'LDA'
alukpts = 4
alunb = 4
alupw = 200
alua = 4.043

# Part 1.2: aluminum dieletric response calculation
aluq_c = np.array([1 / 4., 0., 0.])
alufrq_w = np.linspace(0, 24, 241)
aluKxc = 'ALDA'
aluecut = 50
alueta = 0.05
hilbert = False


# Part 2: iron
# Part 2.1: iron ground state calculation
ironxc = 'LDA'
ironkpts = 4
ironnb = 6
ironpw = 300
irona = 2.867
mm = 2.21

# Part 2.2: iron magnetic response calculation
ironq_c = np.array([0., 0., 1 / 4.])
ironfrq_w = np.linspace(0.300, 0.500, 26)
ironKxc = 'ALDA'
ironecut = 300
ironeta = 0.01

# --------------- Script --------------- #

# Part 1: aluminum
# Part 1.1: aluminum ground state calculation

t1 = time.time()

atoms = bulk('Al', 'fcc', a=alua)
atoms.center()
calc = GPAW(mode=PW(alupw),
            nbands=alunb,
            kpts=(alukpts, alukpts, alukpts),
            parallel={'band': 1},
            idiotproof=False,  # allow uneven distribution of k-points
            xc=aluxc)

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Al', 'all')

t2 = time.time()

# Part 1.2: aluminum dieletric response calculation

df = DielectricFunction(calc='Al',
                        frequencies=alufrq_w,
                        eta=alueta,
                        ecut=aluecut,
                        hilbert=hilbert,
                        disable_spincons_time_reversal=False)
df.get_eels_spectrum(q_c=aluq_c,
                     xc=aluKxc,
                     filename='EELS_Al_ALDA_dsctr«False».csv')

df = DielectricFunction(calc='Al',
                        frequencies=alufrq_w,
                        eta=alueta,
                        ecut=aluecut,
                        hilbert=hilbert,
                        disable_spincons_time_reversal=True)
df.get_eels_spectrum(q_c=aluq_c,
                     xc=aluKxc,
                     filename='EELS_Al_ALDA_dsctr«True».csv')

t3 = time.time()

parprint('Aluminum ground state calculation took', (t2 - t1) / 60, 'minutes')
parprint('Aluminum excited state calculation took', (t3 - t2) / 60, 'minutes')

world.barrier()

# Part 1.3: identify plasmon peaks

d1 = np.loadtxt('EELS_Al_ALDA_dsctr«False».csv', delimiter=',')
d2 = np.loadtxt('EELS_Al_ALDA_dsctr«True».csv', delimiter=',')

wpeak1, Ipeak1 = findpeak(d1[:, 0], d1[:, 2])  # eV
wpeak2, Ipeak2 = findpeak(d2[:, 0], d2[:, 2])

# Part 1.4: compare plasmon peaks

equal(wpeak1, wpeak2, 0.02)


# Part 2: iron
# Part 2.1: iron ground state calculation

t1 = time.time()

Febcc = bulk('Fe', 'bcc', a=irona)
Febcc.set_initial_magnetic_moments([mm])

calc = GPAW(xc=ironxc,
            mode=PW(ironpw),
            kpts=monkhorst_pack((ironkpts, ironkpts, ironkpts)),
            nbands=ironnb,
            idiotproof=False,
            symmetry={'point_group': False},
            parallel={'band': 1})

Febcc.set_calculator(calc)
Febcc.get_potential_energy()
calc.write('Fe', 'all')

t2 = time.time()

# Part 2.2: iron magnetic response calculation

tms = TransverseMagneticSusceptibility(calc='Fe',
                                       frequencies=ironfrq_w,
                                       eta=ironeta,
                                       ecut=ironecut,
                                       disable_spincons_time_reversal=False)

tms.get_dynamic_susceptibility(q_c=ironq_c,
                               xc=ironKxc,
                               filename='iron_dsus_dsctr«False».csv')

tms = TransverseMagneticSusceptibility(calc='Fe',
                                       frequencies=ironfrq_w,
                                       eta=ironeta,
                                       ecut=ironecut,
                                       disable_spincons_time_reversal=True)

tms.get_dynamic_susceptibility(q_c=ironq_c,
                               xc=ironKxc,
                               filename='iron_dsus_dsctr«True».csv')

t3 = time.time()

parprint('Iron ground state calculation took', (t2 - t1) / 60, 'minutes')
parprint('Iron excited state calculation took', (t3 - t2) / 60, 'minutes')

world.barrier()

# Part 2.3: identify magnon peaks

d1 = np.loadtxt('iron_dsus_dsctr«False».csv', delimiter=', ')
d2 = np.loadtxt('iron_dsus_dsctr«True».csv', delimiter=', ')

wpeak1, Ipeak1 = findpeak(d1[:, 0], - d1[:, 4])  # eV
wpeak2, Ipeak2 = findpeak(d2[:, 0], - d2[:, 4])

mw1 = (wpeak1 + d1[0, 0]) * 1000  # meV
mw2 = (wpeak2 + d2[0, 0]) * 1000

# Part 1.4: compare magnon peaks

equal(mw1, mw2, ironeta * 100)
