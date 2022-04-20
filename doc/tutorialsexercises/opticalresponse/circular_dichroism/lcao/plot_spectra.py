# web-page: spectra.png
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6 / 1.62))

for basis, dpath, ls in [('aug.dzp', '.', 'k'), ('dzp', 'dzp', 'k:')]:
    data_ej = np.loadtxt(f'{dpath}/rot_spec.dat')
    ax.plot(data_ej[:, 0], data_ej[:, 1], ls, label=basis)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title(r'Rotatory strength of (R)-methyloxirane')
plt.xlabel('Energy (eV)')
plt.legend(loc='upper left', title='LCAO basis')
plt.ylabel(r'R (10$^{-40}$ cgs eV$^{-1})$')
plt.xlim(0, 10)
plt.ylim(-80, 80)
plt.tight_layout()
plt.savefig('spectra.png')
