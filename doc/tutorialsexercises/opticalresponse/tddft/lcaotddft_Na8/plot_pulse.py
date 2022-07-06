# web-page: pulse.png
import numpy as np
import matplotlib.pyplot as plt
from gpaw.tddft.units import au_to_fs

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 1, 1)

data_ti = np.loadtxt('dmpulse.dat')
ax.plot(data_ti[:, 0] * au_to_fs, data_ti[:, 2], 'k')
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylabel('Dipole moment (atomic units)')
plt.ylim(np.array([-1, 1]) * np.max(np.abs(plt.ylim())))
plt.xlabel('Time (fs)')
plt.xlim(0, 30)

ax = ax.twinx()
data_ti = np.loadtxt('pulse.dat')
ax.plot(data_ti[:, 0] * au_to_fs, data_ti[:, 1], 'g')
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('right')
ax.tick_params(axis='y', labelcolor='g')
plt.ylabel('Pulse (atomic units)', color='g')
plt.ylim(np.array([-1, 1]) * np.max(np.abs(plt.ylim())))

plt.tight_layout()
plt.savefig('pulse.png')
