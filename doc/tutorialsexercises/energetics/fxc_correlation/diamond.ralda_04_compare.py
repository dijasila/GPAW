import numpy as np
from gpaw.test import equal

results = np.loadtxt('diamond.ralda_comparison.dat')

PBE = results[0]
HF = results[1]
RPA = results[2]
rAPBE = results[3]

equal(PBE, -7, 0.01)
equal(HF, -5, 0.01)
equal(RPA, -5, 0.01)
equal(rAPBE, -5, 0.01)
