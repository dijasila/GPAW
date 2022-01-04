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