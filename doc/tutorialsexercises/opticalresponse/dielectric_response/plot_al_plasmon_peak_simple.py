# web-page: al-plasmon-peak.png
import numpy as np
import matplotlib.pyplot as plt

data = np.load('al-plasmon-peak.npz')
fig, ax = plt.subplots()

# Main plot
ax.plot(data['N_k'], data['peak_ik'][0], label='Tetra')
ax.plot(data['N_k'], data['peak_ik'][1], label='Point')
ax.set_xlabel('$N_k^{1/3}$')
ax.set_ylabel('$\\omega_{plasmon}$')
ax.set_xlim([8, 80])
ax.legend(loc=1)
plt.tight_layout()

# Zoom Inset
axins = ax.inset_axes([0.3, 0.5, 0.47, 0.47])
axins.plot(data['N_k'], data['peak_ik'][0], label='Tetra')
axins.plot(data['N_k'], data['peak_ik'][1], label='Point')
axins.set_xlim(32, 80)
axins.set_ylim(data['peak_ik'][0][-1] - 0.25, data['peak_ik'][0][-1] + 0.25)
ax.indicate_inset_zoom(axins)

plt.savefig('al-plasmon-peak.png', bbox_inches='tight')
plt.show()
