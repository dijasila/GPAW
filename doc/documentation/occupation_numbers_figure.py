# creates: occupation_numbers.png
import numpy as np
import matplotlib.pyplot as plt

from gpaw.occupations import fermi_dirac, marzari_vanderbilt, methfessel_paxton

width = 0.05
x = np.linspace(-0.2, 0.2, 101)

plt.plot(x, fermi_dirac(x, 0.0, width)[0],
         label='fermi-dirac')
plt.plot(x, marzari_vanderbilt(x, 0.0, width)[0],
         label='marzari-vanderbilt')
plt.plot(x, methfessel_paxton(x, 0.0, width, 0)[0],
         label='methfessel_paxton-0')
plt.plot(x, methfessel_paxton(x, 0.0, width, 1)[0],
         label='methfessel_paxton-1')
plt.legend()
# plt.show()
plt.savefig('occupation_numbers.png')
