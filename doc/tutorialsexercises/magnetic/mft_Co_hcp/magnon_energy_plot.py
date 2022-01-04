"""Plot magnon energy as a function of rc for all
high-symmetry points of Co(hcp)"""

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import json

# ----- Load results ----- #

# From 'high_sym_pts.py'
rc_r = np.load('rc_r.npy')
E_rmq = np.load('high_sym_pts_E_rmq.npy')
with open('spts.json') as file:
    spts = json.load(file)

# Get info
Nr, N_sites, Nq = E_rmq.shape
qnames = list(spts.keys())

# ----- Plot results ----- #

# Increase plot font
plt.rcParams['font.size'] = 16

# Convert from eV to meV
E_rmq = E_rmq * 1000

# Plot magnon energies vs. integration sphere radii (rc)
plt.figure()
# Define colours for different q-points
cols_q = ['blue', 'red', 'green', 'purple', 'magenta', 'orange']
for q in range(Nq):
    # Plot as low energy magnon band with solid lines
    plt.plot(rc_r, E_rmq[:, 0, q], label=qnames[q], linestyle='-',
             color=cols_q[q])
    # Plot high energy magnon band with dashed lines
    plt.plot(rc_r, E_rmq[:, 1, q], linestyle='--', color=cols_q[q])
plt.xlabel('rc [Å]')
plt.ylabel('Magnon energy [meV]')
plt.legend(prop={'size':14})
plt.savefig('high_sym_pts_energy_vs_rc.png', bbox_inches='tight')