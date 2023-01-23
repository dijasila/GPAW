import numpy as np
from gpaw.test import equal

results = np.loadtxt('diam_kern.ralda_kernel_comparison.dat')

ralda_wave = results[0]
raldac = results[1]
range_RPA = results[2]
RPA = results[3]

equal(ralda_wave, -9.08, 0.01)
equal(raldac, -9.01, 0.01)
equal(range_RPA, -13.84, 0.01)
equal(RPA, -11.17, 0.01)
