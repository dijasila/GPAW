import pickle
import numpy as np

from ase.parallel import paropen
from ase.lattice import bulk

from gpaw.test import equal
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

ecut_equal = np.array([[5.308, 4.860, 4.724],[8.423, 8.083, 7.989]])
for i, ecut in enumerate([50,100,150]):
    fil = pickle.load(paropen('Si-g0w0_k8_ecut%s_results.pckl' %(ecut), 'r'))
    equal(fil['qp'][0,0,1], ecut_equal[1,i], 0.001)
    equal(fil['qp'][0,0,0], ecut_equal[0,i], 0.001)

freq_equal = np.array([8.80138, 8.76186, 8.77281, 8.77516, 8.77466, 8.77462])
for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
    fil = pickle.load(paropen('Si_g0w0_domega0_0.02_omega2_%s_results.pckl' %(omega2),'r'))
    equal(fil['qp'][0,0,1], freq_equal[j], 0.001)

