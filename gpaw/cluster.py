"""Extensions to the ase Atoms class

"""
import numpy as np

from ase import Atoms
from ase.build.connected import connected_indices
from ase.utils import deprecated

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

    @deprecated(
        'Please use connected_indices from ase.build.connected instead.')
    def find_connected(self, index, dmax=None, scale=1.5):
        """Find atoms connected to self[index] and return them."""
        return self[connected_indices(self, index, dmax, scale)]

    @deprecated('Please use adjust_cell from gpaw.cluster instead.')
    def minimal_box(self, border=4, h=None, multiple=4) -> None:
        adjust_cell(self, border, h, multiple)


def adjust_cell(atoms: Atoms, border: float,
                h: float = 0.2, idiv: int = 4) -> None:
    """Adjust the cell such that
    1. The vacuum around all atoms is at least border
       in non-periodic directions
    2. The grid spacing chosen by GPAW will be as similar
       as possible to h in all directions
    """
    n_pbc = atoms.pbc.sum()
    if n_pbc == 3:
        return

    pos_ac = atoms.get_positions()
    lowest_c = np.minimum.reduce(pos_ac)
    largest_c = np.maximum.reduce(pos_ac)

    for i, v_c, in enumerate(atoms.cell):
        if (v_c == 0).all():
            assert not atoms.pbc[i]  # pbc with zero cell size make no sense
            atoms.cell[i, i] = 1

    if n_pbc:
        N_c = h2gpts(h, atoms.cell, idiv)
        gd = GridDescriptor(N_c, atoms.cell, atoms.pbc)
        h_c = gd.get_grid_spacings()
        h = 0
        for pbc, h1 in zip(atoms.pbc, h_c):
            if pbc:
                h += h1 / n_pbc

    # the optimal h to be set to non-periodic directions
    h_c = np.array([h, h, h])

    shift_c = np.zeros(3)

    # adjust each cell direction
    for i in range(3):
        if atoms.pbc[i]:
            continue

        # cell direction
        u_c = atoms.cell[i] / np.linalg.norm(atoms.cell[i])

        extension = (largest_c - lowest_c) * u_c
        min_size = extension + 2 * border

        h = h_c[i]
        N = min_size / h
        N = -(N // -idiv) * idiv  # ceil div
        size = N * h

        atoms.cell[i] = size * u_c

        # shift structure to the center
        shift_c += (size - extension) / 2 * u_c
        shift_c -= lowest_c * u_c

    atoms.translate(shift_c)
