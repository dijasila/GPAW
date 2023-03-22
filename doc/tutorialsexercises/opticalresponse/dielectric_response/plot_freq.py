# web-page: nl_freq_grid.png
import matplotlib.pyplot as plt
import numpy as np
from ase.units import Ha

from gpaw.response.frequencies import NonLinearFrequencyDescriptor

omegamax = 50.0
domega0 = 0.2

plt.figure(figsize=(5, 5))
for omega2 in [2.5, 5, 10, 15, 20, np.inf]:
    x = NonLinearFrequencyDescriptor(
        domega0 / Ha, omega2 / Ha, omegamax / Ha).omega_w * Ha
    y = range(len(x))
    if omega2 == np.inf:
        label = '$\\omega_2 = \\infty$'
    else:
        label = '$\\omega_2 = %.1f\\, \\mathrm{eV}$' % omega2
    plt.plot(x, y, '.', label=label)
plt.ylabel('Freq. no')
plt.xlabel(r'$\omega\, [\mathrm{eV}]$')
plt.axis(xmin=0, xmax=30, ymin=0, ymax=200)
plt.title(r'$\Delta\omega_0 = 0.2\, \mathrm{eV}$')
plt.legend(loc=2)
plt.savefig('nl_freq_grid.png', bbox_inches='tight')
