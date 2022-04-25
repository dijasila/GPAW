# web-page: cu.png
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

fig, ax = plt.subplots(constrained_layout=True)

e0 = None
k = np.arange(8, 21, dtype=float)
for name in ['ITM', 'TM', 'FD-0.05', 'MV-0.2']:
    energies = []
    for n in k:
        e = read(f'Cu-{name}-{int(n)}.txt').get_potential_energy()
        energies.append(e)
    if e0 is None:
        e0 = e
    ax.plot(k**-2, (np.array(energies) - e0) * 1000, label=name)

ax.set_xlabel(r'$1/k^2$')
ax.set_ylabel(r'$\Delta E$ [meV]')
ax2 = ax.secondary_xaxis('top', functions=(lambda x: (x + 1e-10)**-0.5,
                                           lambda k: (k + 1e-10)**-2))
ax2.set_xlabel('Number of k-points (k)')
plt.legend()
plt.savefig('cu.png')
# plt.show()
