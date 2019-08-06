# creates: water.png
import matplotlib.pyplot as plt
import numpy as np

n_m = np.array([32, 64, 128, 256, 384, 576])
#4
dm = np.array([22, 60, 278, 1756, 5206, 17185])
#2
dm_ui = np.array([20, 48, 189, 1092, 3186, 10493])
#3
scf = np.array([29, 69, 306, 2078, 6171, 20950])

f = plt.figure(figsize=(6, 4), dpi=240)
plt.plot(n_m, scf/dm, 'ro-',label='direct min, ss')
plt.plot(n_m, scf/dm_ui, 'bo-',label='direct min, uinv')
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.legend()
plt.ylabel(r'$T_{scf}$ / $T_{dm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1, 2.1)
f.savefig("water.png", bbox_inches='tight')
