# web-page: c2cu.png
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

fig, ax = plt.subplots()

for xc in ['LDA', 'vdW-DF', 'PBE']:
    data = []
    for p in Path().glob(f'{xc}-*.txt'):
        atoms = read(p)
        d = atoms.positions[-1, 2] - atoms.positions[3, 2]
        e = atoms.get_potential_energy()
        data.append((d, e / 2))
    x, y = np.array(sorted(data)).T
    ax.plot(x, y - y[-1], 'x-', label=xc)

data = []
for p in Path().glob('RPA-*.result'):
    d, e, _, _ = (float(x) for x in p.read_text().split())
    data.append((d, e / 2))
x, y = np.array(sorted(data)).T
ax.plot(x, y - y[-1], 'o-', label='RPA')
ax.set_xlim(right=7)
ax.set_ylim(top=0.3)
ax.set_xlabel('distance [Å]')
ax.set_ylabel('energy [eV/C-atom]')
plt.legend()
plt.savefig('c2cu.png')
