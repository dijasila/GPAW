# web-page: traces-Au111.png

import os
import numpy as np
from matplotlib import pyplot
import ase.io

fig, ax = pyplot.subplots(figsize=(4.6, 2.0))
fig.subplots_adjust(bottom=0.2, top=0.99, left=0.15, right=0.97)

data = np.loadtxt(os.path.join('sjm_traces.out', 'cavity.txt'),
                  delimiter=' ')
ax.plot(data[:, 0], data[:, 1], label='solvent')
data = np.loadtxt(os.path.join('sjm_traces.out', 'background_charge.txt'),
                  delimiter=' ')
ax.plot(data[:, 0], data[:, 1], ':', label='jellium')

# Also add atom dots.
atoms = ase.io.read('Au111.traj')
for atom in atoms:
    ax.plot(atom.z, 0.5, 'k.')

ax.set_xlabel('$z$')
ax.set_ylabel('$xy$-averaged value')
ax.legend()
fig.savefig('traces-Au111.png')
