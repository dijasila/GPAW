import numpy as np
from gpaw.test import equal

results = np.loadtxt('range_results.dat')

rc_05 = results[1, 1]
rc_10 = results[2, 1]
rc_20 = results[3, 1]
rc_30 = results[4, 1]
rc_40 = results[5, 1]

equal(rc_05, -12.11, 0.01)
equal(rc_10, -12.04, 0.01)
equal(rc_20, -12.22, 0.01)
equal(rc_30, -12.55, 0.01)
equal(rc_40, -12.84, 0.01)
