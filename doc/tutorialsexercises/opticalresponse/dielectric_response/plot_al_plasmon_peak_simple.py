# web-page: aluminum_EELS.png
import numpy as np
import matplotlib.pyplot as plt

data = np.load('al-plasmon-peak.npz')
plt.plot(data['N_k'], data['peak_ik'][0], label='Tetra')
plt.plot(data['N_k'], data['peak_ik'][1], label='Point')
plt.xlabel('$N_k^{1/3}$')
plt.ylabel('$\\omega_{plasmon}$')
plt.legend()
plt.tight_layout()
plt.savefig('aluminum_EELS.png', bbox_inches='tight')
plt.show()
