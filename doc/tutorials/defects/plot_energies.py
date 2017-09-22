# creates: energies.png
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

data = np.load('formation_energies.npz')

repeat = data['repeats']
uncorrected = data['uncorrected']
line = np.polyfit(1/ repeat[1:], uncorrected[1:], deg=1)
f = np.poly1d(line)
points = np.linspace(0, 1, 100)
corrected = data['corrected']
plt.plot(1 / repeat, uncorrected, 'o', label='No corrections')
plt.plot(1 / repeat, corrected, 'p', label='FNV corrected')
plt.plot(points, f(points), "--")
plt.axhline(line[1], linestyle="dashed")

plt.xlabel('Supercell size', fontsize=18)
plt.ylabel('Energy difference (eV)', fontsize=18)
plt.xlim(xmin=0.06)
plt.xticks(1/ repeat, [str(x) for x in repeat])
plt.legend(loc='lower left')
plt.savefig('energies.png')
