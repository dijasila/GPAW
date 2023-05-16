import matplotlib.pyplot as plt

from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from gpaw.mpi import world
from gpaw.kpt_descriptor import kpts2sizeandoffsets
from gpaw.bztools import find_high_symmetry_monkhorst_pack
from ase.build import bulk
import numpy as np
from ase.dft.kpoints import monkhorst_pack


def get_plasmon_peak(df, q_c):
    _, eels_w = df.get_eels_spectrum(q_c=q_c, filename=None)
    plasmonpeak = np.argmax(eels_w)
    omega = df.get_frequencies()[plasmonpeak]
    return omega


# Part 1: Ground state calculation
atoms = bulk('Al', 'fcc', a=4.043)  # generate fcc crystal structure

# GPAW calculator initialization:
kpts = [8, 8, 8]
calc = GPAW(mode='pw', kpts={'size': kpts, 'gamma': True}, txt=None)

atoms.calc = calc
atoms.get_potential_energy()  # ground state calculation is performed
calc.write('gs-al-peak.gpw')  # use 'all' option to write wavefunctions

peak_ki = []
N_k = []
for i in range(1, 11, 1):
    #kpts = find_high_symmetry_monkhorst_pack('gs-al-peak.gpw', density=i)
    
    N_k.append(i*8)
    responseGS = GPAW('gs-al-peak.gpw').fixed_density(kpts={'size': [i*8, i*8, i*8], 'gamma': True}, txt=None)
    responseGS.write('res-al-peak.gpw', 'all')
    
    # Momentum transfer, must be the difference between two kpoints:
    q_c = [0., 0., 0.]
    
    # Part 2: Spectrum calculation
    df_tetra = DielectricFunction(calc='res-al-peak.gpw',
                                  rate=0.2,
                                  integrationmode='tetrahedron integration')
    tetra_peak = get_plasmon_peak(df_tetra, q_c)

    df = DielectricFunction(calc='res-al-peak.gpw', rate=0.2)#, txt = 'df3.out',)
    point_peak = get_plasmon_peak(df, q_c)
    
    peak_ki.append([tetra_peak, point_peak])

peak_ik = np.array(peak_ki).T
np.savez('al-plasmon-peak.npz', N_k=N_k, peak_ik=peak_ik)
