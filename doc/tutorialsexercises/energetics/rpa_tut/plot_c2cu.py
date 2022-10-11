from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

fig, ax = plt.subplot()

for xc in ['LDA', 'vdW-DF', 'PBE']:
    data = []
    for p in Path().glob(f'{xc}-*.txt'):
        atoms = read(p)
        d = atoms.positions[-1, 2] - atoms.positions[3, 2]
        e = atoms.get_potential_energy()
        data.append((d, e))
    x, y = np.array(sorted(data))
    ax.plot(x, y - y[-1], label=xc)

data = []
for p in Path().glob('RPA-*.txt'):
    d, e, _, _ = (float(x) for x in p.read_text().split())
    data.append((d, e))
x, y = np.array(sorted(data))
ax.plot(x, y - y[-1], label='RPA')
plt.legend()

plt.show()
