import numpy as np
from ase import Atoms
from ase.build import bulk
from gpaw import GPAW, PW
from gpaw.bztools import find_high_symmetry_monkhorst_pack
from gpaw.mpi import size

from gpaw.response.df import DielectricFunction
from gpaw.response.df import read_response_function

assert size <= 4**3

# Ground state calculation

calc = GPAW(mode=PW(200),
            nbands=4,
            setups = {'Na': '1'},
            kpts=kpts1,
            parallel={'band': 1},
            xc='LDA')

atoms.calc = calc
atoms.get_potential_energy()
calc.write('Na_gs')

# Generate grid compatible with tetrahedron integration
kpts = find_high_symmetry_monkhorst_pack('Na_gs', density)

# Calculate the wave functions on the new kpts grid
calc = GPAW('Na_gs').fixed_density(kpts=kpts)
# calc.get_potential_energy()
calc.write('Na', 'all')


# Excited state calculation
q0_c = np.array([0., 0., 0.])
q1_c = np.array([1 / 4., 0., 0.])
w_w = np.linspace(0, 24, 241)

# Calculate the eels spectrum using point integration at both q-points
df1 = DielectricFunction(calc='Na', frequencies=w_w, eta=0.2, ecut=50,
                         hilbert=False, rate=0.2)
df1.get_eels_spectrum(xc='RPA', filename='EELS_Na-PI_q0', q_c=q0_c)
df1.get_eels_spectrum(xc='RPA', filename='EELS_Na-PI_q1', q_c=q1_c)

# Calculate the eels spectrum using tetrahedron integration at q=0
# NB: We skip the finite q-point, because the underlying symmetry
# exploration runs excruciatingly slowly at finite q...
df2 = DielectricFunction(calc='Na', eta=0.2, ecut=50,
                         integrationmode='tetrahedron integration',
                         hilbert=True, rate=0.2)
df2.get_eels_spectrum(xc='RPA', filename='EELS_Na-TI_q0', q_c=q0_c)

omegaP0_w, eels0P0_w, eelsP0_w = read_response_function('EELS_Na-PI_q0')
omegaP1_w, eels0P1_w, eelsP1_w = read_response_function('EELS_Na-PI_q1')
omegaT0_w, eels0T0_w, eelsT0_w = read_response_function('EELS_Na-TI_q0')

# import matplotlib.pyplot as plt
# plt.subplot(1, 2, 1)
# plt.plot(omegaP0_w, eelsP0_w)
# plt.plot(omegaT0_w, eelsT0_w)
# # plt.subplot(1, 2, 2)
# # plt.plot(omegaP1_w, eelsP1_w)
# plt.xlim((3,9))
# plt.show()

