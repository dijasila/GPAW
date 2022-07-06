# web-page: Ag55_spec_basis.png
import numpy as np
import matplotlib.pyplot as plt

data_ej = np.loadtxt('spec.dat')
data_my_ej = np.loadtxt('mybasis/spec.dat')
data_dzp_ej = np.loadtxt('dzp/spec.dat')

plt.figure(figsize=(6, 6 / 1.62))
ax = plt.subplot(1, 1, 1)
ax.plot(data_ej[:, 0], data_ej[:, 1], 'k', label='p-valence')
ax.plot(data_my_ej[:, 0], data_my_ej[:, 1], 'k--', label='p-valence (my)')
ax.plot(data_dzp_ej[:, 0], data_dzp_ej[:, 1], 'k:', label='dzp')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title(r'Effect of LCAO basis on the spectrum')
plt.xlabel('Energy (eV)')
plt.legend(loc='upper left', title='LCAO basis')
plt.ylabel('Photoabsorption (eV$^{-1}$)')
plt.xlim(0, 6)
plt.ylim(ymin=0)
plt.tight_layout()
plt.savefig('Ag55_spec_basis.png')
