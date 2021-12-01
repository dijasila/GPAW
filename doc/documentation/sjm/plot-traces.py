# web-page: traces.png
import pickle
import numpy as np
import ase.io
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib import pyplot
from matplotlib.patches import Circle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


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


def get_potential_trace(filename):
    with open(filename, 'rb') as f:
        phi = pickle.load(f)
    phi = phi.mean(axis=(0, 1))
    return phi


fig, ax_potential = pyplot.subplots(figsize=(6.4, 2.2))
fig.subplots_adjust(bottom=0.2, top=0.99, left=0.11, right=0.90)
atoms = ase.io.read('atoms.traj')

###########################
# Potential axis.
###########################

z_trace44 = get_potential_trace('esp4.4V.pckl')
z_trace43 = get_potential_trace('esp4.3V.pckl')
z_trace44 -= z_trace44[0]
z_trace43 -= z_trace43[0]
diff = z_trace43 - z_trace44
zs = np.linspace(0., atoms.cell[2][2], num=len(diff))
ax_potential.plot(zs, diff, color='C1', linewidth=2)
label_at = int(len(zs) * 0.80)
ax_potential.text(zs[label_at], diff[label_at] + 0.01, 'potential',
                  ha='center', va='center', color='C1')

###########################
# Electrons axis.
###########################

ax_electrons = ax_potential.twinx()
# ax1.set_zorder(ax2.get_zorder()+1)

z_trace44 = get_electron_trace('all4.4V.pckl')
z_trace43 = get_electron_trace('all4.3V.pckl')
diff = z_trace43 - z_trace44
ax_electrons.plot(zs, diff)
ax_electrons.text(zs[label_at], diff[label_at] - 0.003, 'electrons',
                  ha='center', va='center', color='C0')

###########################
# Atoms axis.
###########################

ax_atoms = ax_potential.twinx()

# Sort atoms by x value.
cell = atoms.cell
atomslist = [atom for atom in atoms]
atomslist = sorted(atomslist, key=lambda atom: atom.x)
atomslist.reverse()

# Add the atoms to the plot as circles.
for atom in atomslist:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    circle = Circle((atom.z, atom.y), radius, facecolor='none',
                    edgecolor='0.5', linewidth=0.5)
    ax_atoms.add_patch(circle)

###########################
# Touch up.
###########################

ax_atoms.axis('equal')
ax_potential.set_xlabel(r'$z$ coordinate, $\mathrm{\AA}$')
ax_atoms.set_yticks([])

ax_potential.set_zorder(2)
ax_electrons.set_zorder(1)
ax_potential.patch.set_visible(False)

ax_potential.set_ylabel(r'$\phi_{xy}$(4.4 V) - $\phi_{xy}$(4.3 V), V')
ax_potential.yaxis.set_major_locator(MultipleLocator(0.1))
ax_potential.yaxis.set_minor_locator(AutoMinorLocator(5))
ax_electrons.set_ylabel(r'$n_{xy}$(4.4 V) - $n_{xy}$(4.3 V)')
ax_electrons.yaxis.set_major_locator(MultipleLocator(0.01))
ax_electrons.yaxis.set_minor_locator(AutoMinorLocator(2))

fig.savefig('traces.png')
