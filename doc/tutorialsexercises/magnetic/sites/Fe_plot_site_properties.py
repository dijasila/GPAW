"""Plot the site magnetization and site spin splitting of Fe(bcc)."""

import numpy as np
import matplotlib.pyplot as plt

# Load data
rc_r = np.load('rc_r.npy')
m_r = np.load('m_r.npy')
dxc_r = np.load('dxc_r.npy')

# Plot data
rlabel = r'$r_\mathrm{c}$ [$\mathrm{\AA}$]'
plt.subplot(1, 2, 1)
plt.plot(rc_r, m_r)
plt.xlabel(rlabel)
plt.ylabel(r'Site magnetization [$\mu_\mathrm{B}$]')
plt.subplot(1, 2, 2)
plt.plot(rc_r, dxc_r)
plt.xlabel(rlabel)
plt.ylabel('Site spin splitting [eV]')

# Save as png
filename = 'Fe_site_properties.png'
plt.savefig(filename, format='png', bbox_inches='tight')
