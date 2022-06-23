"""Plot dispersion relation for Co(hcp) between all high-symmetry points"""

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import json

# ----- Plotting functions ----- #

def magnon_dispersion_plot(ax, q_qc, E_mq, spts=None):
    """Plot magnon dispersion relation.
    If dictionary of special, named points is given, set tickmarks
    at corresponding simulated points.

    Parameters
    ----------
    ax : matplotlib axis object to populate
    q_qc : nd.array
        simulated q-points
    E_mq : nd.array
        magnon energies
    spts : dict
        name (key) and coordinates (value) of special, named q-points
        e.g. high-symmetry points

    """

    # ----- Initial calculation ----- #

    # Get number of q-points and sites
    N_sites, Nq = E_mq.shape

    # Get distance along path for each q-point
    # Differences between subsequent q-points
    qdiff_q = q_qc[1:, :] - q_qc[:-1, :]
    # Distances between subsequent q-points![](Magnon_dispersion_Co_hcp.png)
    qdist_q = np.sqrt(np.sum(qdiff_q ** 2, axis=-1))
    pathq = [0]  # Summed distance along path (start at 0)
    for qdist in qdist_q:
        pathq += [pathq[-1] + qdist]

    # Get tickmark positions and values (point names)
    if spts is None:
        x_ticks_vals, x_ticks_pos = [], []
    else:
        x_ticks_vals, x_ticks_pos = get_tickmarks(q_qc, pathq, spts)

    # ----- Do plot ----- #

    # Plot dispersion lines
    for m in range(N_sites):
        E_q = E_mq[m, :]
        ax.plot(pathq, E_q)
        ax.plot(pathq, E_q, marker='x')
    ax.set_xticks(ticks=x_ticks_pos)
    ax.set_xticklabels(labels=x_ticks_vals)
    for x in x_ticks_pos:
        ax.axvline(x=x, color='steelblue')

def get_tickmarks(q_qc, pathq, spts):
    """Get tickmark positions and values (point names)
    Determines if any of the simulated points are identical to a special named
    point. If yes, use name as tick value.

    Parameters
    ----------
    q_qc : nd.array
        simulated points
    pathq : iterable
        positions of simulated points along path
    spts : dict
        name (key) and coordinates (value) of special, named q-points
        e.g. high-symmetry points
    """

    qnames = list(spts.keys())
    qspc_qc = np.vstack([spts[key] for key in qnames])

    # Find if any simulated q-points match the special points
    x_ticks_vals, x_ticks_pos = [], []
    for q, q_c in enumerate(q_qc):
        # Compare q_c with all special points
        is_close = np.all(np.isclose(q_c, qspc_qc), axis=-1)
        # If single match; find name of matching point and position along path
        if np.sum(is_close) == 1:
            qind = np.argwhere(is_close)[0, 0]  # Index of special point
            x_ticks_pos += [pathq[q]]
            x_ticks_vals += [qnames[qind]]

    return x_ticks_vals, x_ticks_pos

# ----- Load results ----- #

# From 'high_sym_path.py'
q_qc = np.load('q_qc.npy')
E_mq = np.load('high_sym_path_E_mq.npy')

# From 'high_sym_pts.py'
with open('spts.json') as file:
    spts = json.load(file)
# Turn lists back into arrays
for key in spts.keys():
    spts[key] = np.array(spts[key])

# Get info
N_sites, Nq = E_mq.shape

# ----- Plot results ----- #

# Increase font size in plots
plt.rcParams['font.size'] = 16

# Convert from eV to meV
E_mq = E_mq * 1000

# Plot dispersion relation
fig, ax = plt.subplots()
magnon_dispersion_plot(ax, q_qc, E_mq, spts=spts)
ax.set_ylabel(f'Energy [meV]')
ax.set_title(f'Magnon dispersion for Co(hcp)')
savename = 'Magnon_dispersion_Co_hcp.png'
plt.savefig(savename, bbox_inches='tight')
print(f'Magnon dispersion plot saved in {savename}')