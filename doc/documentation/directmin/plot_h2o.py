# creates: water.png
import matplotlib.pyplot as plt
import numpy as np

f = plt.figure(figsize=(12, 4), dpi=240)
plt.subplot(121)

n_m = np.array([32, 64, 128])

# see data from wm_scf.py and wm_dm.py
dm_ui = np.array([13, 47, 243])
scf = np.array([22, 91, 659])

plt.title('Ratio of total elapsed times')
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.ylabel(r'$T_{scf}$ / $T_{etdm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1.0, 3.0)
plt.yticks(np.arange(1, 3.1, 0.5))
plt.plot(n_m, scf / dm_ui, 'bo-')

plt.subplot(122)
# see data from wm_scf.py and wm_dm.py
# add 2 because it also performs diagonalization
# in the begining and the end of etdm
dm_ui = np.array([13/(15 + 2), 47/(15 + 2), 243/(15 + 2)]) 
scf = np.array([22/22, 91/21, 659/21])

plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.title('Ratio of elapsed times per iteration')
plt.ylabel(r'$T_{scf}$ / $T_{etdm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1.0, 3.0)
plt.yticks(np.arange(1, 3.1, 0.5))
plt.plot(n_m, scf / dm_ui, 'ro-')

f.savefig("water.png", bbox_inches='tight')

