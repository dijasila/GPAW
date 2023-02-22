import numpy as np
from gpaw.test import equal

results = np.loadtxt('diam_kern.ralda_kernel_comparison.dat')

ralda_dens = results[0]
ralda_wave = results[1]
RPA = results[2]

equal(ralda_dens, -9.224 0.01)
equal(ralda_wave, -9.08, 0.01)
equal(RPA, -11.17, 0.01)
