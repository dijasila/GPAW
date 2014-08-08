from gpaw.response.chi0 import frequency_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('figure', figsize=(4.0, 4.0), dpi=800)

omegamax = 10
domega0 = 0.1

ex0_w = frequency_grid(domega0, omegamax, 0.0) / omegamax
exp5_w = frequency_grid(domega0, omegamax, 0.5) / omegamax
ex1_w = frequency_grid(domega0, omegamax, 1.0) / omegamax
ex2_w = frequency_grid(domega0, omegamax, 2.0) / omegamax
ex3_w = frequency_grid(domega0, omegamax, 3.0) / omegamax

plt.figure()
plt.plot(ex0_w, np.arange(len(ex0_w)), '.', label='$\\alpha = 0$')
plt.plot(exp5_w, np.arange(len(exp5_w)), '.', label='$\\alpha = 0.5$')
plt.plot(ex1_w, np.arange(len(ex1_w)), '.', label='$\\alpha = 1$')
plt.plot(ex2_w, np.arange(len(ex2_w)), '.', label='$\\alpha = 2$')
plt.plot(ex3_w, np.arange(len(ex3_w)), '.', label='$\\alpha = 3$')
plt.ylabel('Freq. no')
plt.xlabel('$\\omega / \\omega_\mathrm{max}$')
plt.legend(loc=2)
plt.savefig('nl_freq_grid.png', bbox_inches='tight')
