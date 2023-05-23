from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from gpaw.test import findpeak
from ase.build import bulk
import numpy as np


def get_plasmon_peak(df, q_c):
    _, eels_w = df.get_eels_spectrum(q_c=q_c, filename=None)
    omega_w = df.get_frequencies()
    wpeak, _ = findpeak(omega_w, eels_w)
    return wpeak


# Part 1: Ground state calculation
atoms = bulk('Al', 'fcc', a=4.043)  # generate fcc crystal structure

# GPAW calculator initialization:
kpts = [16, 16, 16]
calc = GPAW(mode='pw', kpts={'size': kpts, 'gamma': True}, txt=None)

atoms.calc = calc
atoms.get_potential_energy()  # ground state calculation is performed
calc.write('gs-al-peak.gpw')

peak_ki = []
N_k = []
for i in range(1, 11, 1):
    k = i * 8
    N_k.append(k)
    kpts = {'size': [k, k, k], 'gamma': True}
    responseGS = GPAW('gs-al-peak.gpw').fixed_density(kpts=kpts, txt=None)
    responseGS.write('res-al-peak.gpw', 'all')  # 'all' to write wavefunctions

    # Momentum transfer, must be the difference between two kpoints:
    q_c = [0., 0., 0.]

    # Part 2: Spectrum calculation
    df_tetra = DielectricFunction(calc='res-al-peak.gpw',
                                  rate=0.2,
                                  integrationmode='tetrahedron integration')
    tetra_peak = get_plasmon_peak(df_tetra, q_c)

    df = DielectricFunction(calc='res-al-peak.gpw', rate=0.2)
    point_peak = get_plasmon_peak(df, q_c)

    peak_ki.append([tetra_peak, point_peak])

peak_ik = np.array(peak_ki).T
np.savez('al-plasmon-peak.npz', N_k=N_k, peak_ik=peak_ik)
