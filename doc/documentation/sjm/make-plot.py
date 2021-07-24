# web-page: overview.png

import ase.io
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from matplotlib import pyplot
from matplotlib.patches import Circle

fig, ax_solvent = pyplot.subplots(figsize=(6., 2.2))
fig.subplots_adjust(bottom=0.2, top=0.99, left=0.01, right=0.99)

###########################
# Solvent axis.
###########################

with open('sjm_traces4.4.out/cavity.txt', 'r') as f:
    lines = f.read().splitlines()

zs = []
cavities = []
for line in lines:
    z, cavity = line.split()
    zs.append(float(z))
    cavities.append(float(cavity))

where = [True if cavity != 0 else False for cavity in cavities]
ax_solvent.fill_between(zs, cavities, color='C0', where=where)
ax_solvent.text(17.8, max(cavities) / 2., 'solvent', ha='center',
                va='center', rotation=90.)

###########################
# Jellium axis.
###########################

ax_jellium = ax_solvent.twinx()

with open('sjm_traces4.4.out/background_charge.txt', 'r') as f:
    lines = f.read().splitlines()

zs = []
jelliums = []
for line in lines:
    z, jellium = line.split()
    zs.append(float(z))
    jelliums.append(float(jellium))

where = [True if jellium != 0 else False for jellium in jelliums]
ax_jellium.fill_between(zs, jelliums, where=where, hatch='///', ec='0.2')
ax_jellium.text(24., max(jelliums) / 2., 'jellium', ha='center', va='center')

###########################
# Atoms axis.
###########################

ax_atoms = ax_jellium.twinx()

# Sort atoms by x value.
atoms = ase.io.read('atoms.traj')
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
    ax_atoms.add_patch(circle)

###########################
# Touch up.
###########################

for ax in [ax_jellium, ax_solvent]:
    # Makes it include the left side in all the aspect nonsense.
    ax.plot(0., 0., '.', color='w')

ax_atoms.set_xlabel('$z$ coordinate')
ax_atoms.set_yticks([])
ax_jellium.set_yticks([])
ax_solvent.set_yticks([])

ax_atoms.set_xlim(0., cell[-1][-1])
ax_atoms.axis('equal')
for ax in [ax_jellium, ax_solvent]:
    ax.set_ylim(bottom=0.)

ax_jellium.set_ylim(0., max(jelliums))
ax_solvent.set_ylim(0., max(cavities))

fig.savefig('overview.png')
