# web-page: water.png
import matplotlib.pyplot as plt
import numpy as np

# Data from wm_dm_vs_scf.py
calculated_data = np.genfromtxt('water-results.txt')

# x should be number of water molecules.
# First column is number of atoms, so divide by 3 to
# obtain the number of water molecules.
x = calculated_data[:, 0] / 3

f = plt.figure(figsize=(12, 4), dpi=240)

plt.subplot(121)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.title('Ratio of total elapsed times')
plt.ylabel(r'$T_{scf}$ / $T_{etdm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1.0, 3.0)
plt.yticks(np.arange(1, 3.1, 0.5))
plt.plot(x, calculated_data[:, 1], 'bo-')

plt.subplot(122)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.title('Ratio of elapsed times per iteration')
plt.ylabel(r'$T_{scf}$ / $T_{etdm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1.0, 3.0)
plt.yticks(np.arange(1, 3.1, 0.5))
plt.plot(x, calculated_data[:, 2], 'ro-')

f.savefig("water.png", bbox_inches='tight')
