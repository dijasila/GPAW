from gpaw.lcaotddft.tddfpt import transform_local_operator
from ase.io.cube import read_cube
import matplotlib.pyplot as plt
import numpy as np
transform_local_operator(
    gpw_file='Na8_gs.gpw', tdop_file='Na8.TdDen', fqop_file='Na8.FqDen', omega=1.8, eta=0.23)
data, atoms = read_cube('Na8.FqDen.imag.cube', read_data=True)
data = data[:,:, 16]  #data = np.sum(data, axis=2)
extent = [0,atoms.cell[0][0],0,atoms.cell[1][1]]
plt.imshow(data.T, origin='lower', extent=extent)

for atom in atoms:
    circle=plt.Circle((atom.position[0],atom.position[1]),.3,color='r', clip_on=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)

plt.title('Induced density of $Na_{8}$')
plt.xlabel('$\\AA$')
plt.ylabel('$\\AA$')
plt.savefig('Na8_imag.png')
