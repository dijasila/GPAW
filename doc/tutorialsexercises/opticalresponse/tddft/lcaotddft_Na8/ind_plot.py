# web-page: ind_1.12.png, ind_2.48.png
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from gpaw.mpi import world

assert world.size == 1, 'This script should be run in serial mode.'


def do(freq):
    # Read cube file
    cube = read(f'ind_{freq:.2f}.cube', full_output=True)
    d_g = cube['data']
    atoms = cube['atoms']
    box = np.diag(atoms.get_cell())
    ng = d_g.shape

    # Take slice of data array
    d_yx = d_g[:, :, ng[2] // 2]
    x = np.linspace(0, box[0], ng[0])
    xlabel = u'x (Å)'
    y = np.linspace(0, box[1], ng[1])
    ylabel = u'y (Å)'

    # Plot
    plt.figure(figsize=(8, 3.5))
    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(x, y)
    dmax = max(d_yx.min(), d_yx.max())
    vmax = 0.9 * dmax
    vmin = -vmax
    plt.pcolormesh(X, Y, d_yx.T, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   shading='auto')
    contours = np.sort(np.outer([-1, 1], [0.02]).ravel() * dmax)
    plt.contour(X, Y, d_yx.T, contours, cmap='RdBu_r', vmin=-1e-10, vmax=1e-10)
    for atom in atoms:
        pos = atom.position
        plt.scatter(pos[0], pos[1], s=100, c='k', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([x[0], x[-1]])
    plt.ylim([y[0], y[-1]])
    ax.set_aspect('equal')

    plt.title(f'Induced density of Na8 at {freq:.2f} eV')
    plt.tight_layout()
    plt.savefig(f'ind_{freq:.2f}.png')


do(1.12)
do(2.48)
