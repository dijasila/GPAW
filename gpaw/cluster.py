"""Extensions to the ase Atoms class

"""
import numpy as np

from ase import Atoms
from ase.build.connected import connected_indices

from gpaw.utilities import h2gpts
from gpaw.grid_descriptor import GridDescriptor


class Cluster(Atoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, *args, **kwargs):

        self.data = {}

        if len(args) > 0:
            filename = args[0]
            if isinstance(filename, str):
                self.read(filename, kwargs.get('filetype'))
                return
        else:
            Atoms.__init__(self, [])

        if kwargs.get('filename') is not None:
            filename = kwargs.pop('filename')
            Atoms.__init__(self, *args, **kwargs)
            self.read(filename, kwargs.get('filetype'))
        else:
            Atoms.__init__(self, *args, **kwargs)

    def find_connected(self, index, dmax=None, scale=1.5):
        """Find atoms connected to self[index] and return them."""
        return self[connected_indices(self, index, dmax, scale)]

    def minimal_box(self, border=4, h=None, multiple=4) -> None:
        adjust_cell(self, border, h, multiple)


def adjust_cell(atoms: Atoms, border: float,
                h: float = 0.2, idiv: int = 4) -> None:
    """Adjust the cell such that
    1. The vacuum around all atoms is at least border
       in non-periodic directions
    2. The grid spacing chosen by GPAW will be as similar
       as possible in all directions
    """
    n_pbc = atoms.pbc.sum()
    if n_pbc == 3:
        return
    # extreme positions

    pos_ac = atoms.get_positions()
    lowest_c = np.minimum.reduce(pos_ac)
    largest_c = np.maximum.reduce(pos_ac)

    if (atoms.cell == 0.).any():
        for i in range(3):
            if atoms.pbc[i]:
                continue
            atoms.cell[i, i] = 1

    if n_pbc:
        N_c = h2gpts(h, atoms.cell, idiv)
        gd = GridDescriptor(N_c, atoms.cell, atoms.pbc)
        h_c = gd.get_grid_spacings()
        h = 0
        for pbc, h1 in zip(atoms.pbc, h_c):
            if pbc:
                h += h1 / n_pbc

    # this is the wished h to be set to non-periodic directions
    h_c = np.array([h, h, h])

    shift_c = np.zeros(3)

    # adjust each cell direction
    for i in range(3):
        if atoms.pbc[i]:
            continue

        extension = largest_c[i] - lowest_c[i]
        min_size = extension + 2 * border

        h = h_c[i]
        N = min_size / h
        N = -(N // -idiv) * idiv  # ceil div
        size = N * h

        atoms.cell[i] *= size / np.linalg.norm(atoms.cell[i])

        # shift structure to the center
        shift_c[i] = (size - extension) / 2
        shift_c[i] -= lowest_c[i]

    atoms.translate(shift_c)
