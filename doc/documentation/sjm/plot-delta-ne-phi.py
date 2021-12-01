# web-page: delta-ne-phi.png

import pickle
import numpy as np
from matplotlib import pyplot
import ase.io


def get_potential_trace(filename):
    with open(filename, 'rb') as f:
        phi = pickle.load(f)
    phi = phi.mean(axis=(0, 1))
    return phi


def get_electron_trace(filename):
    with open(filename, 'rb') as f:
        n = pickle.load(f)
    # Multiply each point by its local cell volume to convert from
    # electron density to electrons in each cube.
    vol_adjust = atoms.cell.volume / np.prod(n.shape)
    n *= vol_adjust
    z_trace = np.sum(n, axis=(0, 1))
    z_trace = np.cumsum(z_trace)
    return z_trace


excess_electrons = [-0.2, -0.1, 0., 0.1, 0.2]
atoms = ase.io.read('atoms0.0000.traj')

fig, axes = pyplot.subplots(nrows=2, figsize=(6.4, 4.0), sharex=True)
fig.subplots_adjust(right=0.99, top=0.99)

###########################
# Potential axis.
###########################

ax = axes[0]
z_trace0 = get_potential_trace('esp0.0000.pckl')
z_trace0 -= z_trace0[0]

for excess_electron in excess_electrons:
    label = '{:.4f}'.format(excess_electron)
    z_trace = get_potential_trace(f'esp{label}.pckl')
    z_trace -= z_trace[0]
    diff = z_trace - z_trace0
    zs = np.linspace(0., atoms.cell[2][2], num=len(diff))
    ax.plot(zs, diff, color='C1', linewidth=2)
    ax.text(zs[-1] + 1., diff[-1], '{:+.1f}'.format(excess_electron),
            ha='left', va='center', color='C1')

###########################
# Electrons axis.
###########################

ax = axes[1]
z_trace0 = get_electron_trace('allelectrondensity0.0000.pckl')

for excess_electron in excess_electrons:
    label = '{:.4f}'.format(excess_electron)
    z_trace = get_electron_trace(f'allelectrondensity{label}.pckl')
    diff = z_trace - z_trace0
    ax.plot(zs, diff, color='C0')
    ax.text(zs[-1] + 1., diff[-1], '{:+.1f}'.format(excess_electron),
            ha='left', va='center', color='C0')

###########################
# Touch up.
###########################

axes[0].set_xlim(0., zs[-1] + 4.)
axes[0].set_ylabel(r'$\Delta \phi_{xy}$, V')
axes[1].set_ylabel(r'$\Delta n_{xy}$')
axes[1].set_xlabel(r'$z$ coordinate, $\mathrm{\AA}$')

fig.savefig('delta-ne-phi.png')
