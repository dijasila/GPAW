import numpy as np
from gpaw.test import equal

results = np.loadtxt('results.dat')

diff_222 = results[3]

equal(diff_222, 21.78, 0.01)

potfile = open('model_potentials.dat','r')
El = float(potfile.readline()[12:])

equal(El,-1.28,0.01)
