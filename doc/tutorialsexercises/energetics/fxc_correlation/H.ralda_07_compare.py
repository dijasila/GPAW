import numpy as np
from gpaw.test import equal

results = np.loadtxt('H.ralda_comparison.dat')

LDA = results[0]
RPA_LDA = results[1]
rALDA = results[2]
PBE = results[3]
RPA_PBE = results[4]
rAPBE = results[5]

equal(LDA, -0.56, 0.01)
equal(RPA_LDA, -0.56, 0.01)
equal(rALDA, -0.03, 0.01)
equal(PBE, -0.15, 0.01)
equal(RPA_PBE, -0.56, 0.01)
equal(rAPBE, -0.01, 0.01)
