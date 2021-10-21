# web-page: overview.png

import numpy as np
from ase.io.cube import read_cube
from matplotlib import pyplot
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle


fig, ax = pyplot.subplots(figsize=(6., 1.8))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.99)

###########################
# Plot solvent.
###########################

with open('sjm_traces4.4V-cube.out/cavity.cube') as f:
    data = read_cube(f)

atoms = data['atoms']
cube = data['data']
flat = np.mean(cube, axis=0)

ax.imshow(flat, cmap='Blues', interpolation='spline16',
          origin='lower', vmax=1.5,
          extent=(0., atoms.cell[2, 2], 0., atoms.cell[1, 1]))

###########################
# Add atoms as circles.
###########################

cell = atoms.cell
atomslist = [atom for atom in atoms]
atomslist = sorted(atomslist, key=lambda atom: atom.x)
atomslist.reverse()

# Add the atoms to the plot as circles.
for atom in atomslist:
    color = jmol_colors[atom.number]
    radius = radii[atom.number]
    circle = Circle((atom.z, atom.y), radius, facecolor=color,
                    edgecolor='k', linewidth=1)
    ax.add_patch(circle)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

###########################
# Add the jellium region.
###########################

with open('sjm_traces4.4V.out/background_charge.txt', 'r') as f:
    lines = f.read().splitlines()

zs = []
jelliums = []
for line in lines:
    z, jellium = line.split()
    zs.append(float(z))
    jelliums.append(float(jellium))

jelliums = np.array(jelliums) * atoms.cell[1][1]

where = [True if jellium != 0 else False for jellium in jelliums]
ax.fill_between(zs, jelliums, where=where, hatch='///', ec='0.2')

###########################
# Touch up.
###########################

ax.text(24., max(jelliums) / 2., 'jellium', ha='center', va='center')
ax.text(17.8, max(jelliums) / 2., 'solvent', ha='center',
        va='center', rotation=90.)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel(r'$z$, $\mathrm{\AA}$')
ax.set_ylabel(r'$y$, $\mathrm{\AA}$')

fig.savefig('overview.png')
