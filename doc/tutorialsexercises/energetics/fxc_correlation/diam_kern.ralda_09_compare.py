import numpy as np
from gpaw.test import equal

results = np.loadtxt('diam_kern.ralda_kernel_comparison.dat')

ralda_wave = results[0]
RPA = results[1]

equal(ralda_wave, -9.08, 0.01)
equal(RPA, -11.17, 0.01)
