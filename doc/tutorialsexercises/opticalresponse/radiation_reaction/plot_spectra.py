# web-page: spectra_nad.png
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6 / 1.62))

data = np.loadtxt('spec_nad.dat')
ax.plot(data[:, 0], data[:, 3], label=r'Na$_2$')
data = np.loadtxt('spec_nad2.dat')
ax.plot(data[:, 0], data[:, 3], label=r'2Na$_2$')
plt.xlabel('Energy (eV)')
plt.legend()
plt.ylabel(r'Photoabsorption (1/eV)')
plt.xlim(3., 3.16)
plt.tight_layout()
plt.savefig('spectra_nad.png')
