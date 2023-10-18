"""Plot the site pair spin splitting calculated with varrying number of bands.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
rc_r = np.load('rc_r.npy')
dxc_r = np.load('dxc_r.npy')
sp_dxc_r = np.load('sp_dxc_r.npy')
dxc_nr = np.load('dxc_nr.npy')

# Plot data
fig, ax = plt.subplots()
ax.plot(rc_r, dxc_r)
ax.plot(rc_r, sp_dxc_r, c='0.5', linestyle=':')
for n, unocc in enumerate(4 * np.arange(9)):
    ax.plot(rc_r, dxc_nr[n].real, label=unocc)
ax.legend(title='unocc', loc=0)
ax.set_xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
ax.set_xlim((0., 1.4))
ax.set_ylabel('Site spin splitting [eV]')
ax.set_ylim((0., None))
# Plot data on an inset displaying the flat region
ax2 = ax.inset_axes([0.565, 0.08, 0.41, 0.45])
ax2.plot(rc_r, dxc_r)
ax2.plot(rc_r, sp_dxc_r, c='0.5', linestyle=':')
for n, unocc in enumerate(4 * np.arange(9)):
    ax2.plot(rc_r, dxc_nr[n].real, label=unocc)
ax2.set_xlim((0.791, 1.365))
ax2.set_ylim((5.4, 5.6))

# Save as png
filename = 'Fe_site_sum_rules.png'
plt.savefig(filename, format='png', bbox_inches='tight')
