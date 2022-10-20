import pickle
import numpy as np
from ase.parallel import paropen


ecut_equal = np.array([[19.157, 18.639, 18.502],
                       [11.814, 11.165, 10.974]])
for i, ecut in enumerate([100, 200, 300]):
    fil = pickle.load(paropen(f'C-g0w0_k8_ecut{ecut}_results_GW.pckl', 'rb'))
    assert abs(fil['qp'][0, 0, 1] - ecut_equal[0, i]) < 0.01
    assert abs(fil['qp'][0, 0, 0] - ecut_equal[1, i]) < 0.01

freq_equal = np.array([19.91, 19.80, 19.89, 19.92, 19.90, 19.91])
for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
    fil = pickle.load(paropen('C_g0w0_domega0_0.02_omega2_%s_results_GW.pckl'
                              % omega2, 'rb'))
    assert abs(fil['qp'][0, 0, 1] - freq_equal[j]) < 0.01
