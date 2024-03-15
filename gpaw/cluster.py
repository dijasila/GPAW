"""Extensions to the ase Atoms class

"""
import numpy as np

from ase import Atoms
from ase.io import read
from ase.build.connected import connected_indices
from ase.atoms import string2vector

from gpaw.utilities import h2gpts


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

    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.get_positions()
        return np.array([np.minimum.reduce(pos), np.maximum.reduce(pos)])

    def find_connected(self, index, dmax=None, scale=1.5):
        """Find atoms connected to self[index] and return them."""
        return self[connected_indices(self, index, dmax, scale)]

    def minimal_box(self, border=4, h=None, multiple=4) -> None:
        adjust_cell(self, border, h, multiple)

    def read(self, filename, format=None):
        """Read the structure from some file. The type can be given
        or it will be guessed from the filename."""

        self.__init__(read(filename, format=format))
        return len(self)

    def rotate_cell(self, a: str, v: str, center=(0, 0, 0)):
        '''Rotate molecule and cell so we are left with the indexes
        x,y,z = 0,1,2, the ase atosm.rotate does not conserve these
        indexes of the and when ajusting the box this causes trubles'''

        self.rotate(a, v, center)

        # Rotate the axises into each other
        a = string2vector(a)
        v = string2vector(v)

        v1 = v / np.linalg.norm(v)
        a1 = a / np.linalg.norm(a)

        c = np.dot(v1, a1)
        v1 = np.cross(v1, a1)
        s = np.linalg.norm(v1)

        v1 /= s

        rotcell = self.get_cell()
        rotcell[:] = (c * rotcell -
                      np.cross(rotcell, s * v1) +
                      np.outer(np.dot(rotcell, v1), (1.0 - c) * v1))

        for i in range(3):
            if v[i] != 0:
                iv = i

            if a[i] != 0:
                ia = i

        cell_v = rotcell[ia].copy()
        cell_a = rotcell[iv].copy()

        rotcell[iv] = - cell_v
        rotcell[ia] = - cell_a

        self.set_cell(rotcell)
        pbc = self.pbc
        if pbc[iv] != pbc[ia]:
            pbc[iv], pbc[ia] = pbc[ia], pbc[iv]
        self.pbc = pbc


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
        # logic from gpaw/utilitis/__init__.py
        h_c = (np.linalg.inv(atoms.cell)**2).sum(0)**-0.5 / N_c
        h = 0
        for pbc, h1 in zip(atoms.pbc, h_c):
            if pbc:
                h += h1 / n_pbc

    h_c = np.array([h, h, h])

    shift_c = np.zeros(3)

    # adjust each cell direction
    for i in range(3):
        if atoms.pbc[i]:
            continue

        extension = largest_c[i] - lowest_c[i]
        min_size = extension + 2 * border

        h = h_c[i]
        # logic from gpaw/utilitis/__init__.py
        N = np.maximum(idiv,
                       (min_size / h / idiv + 0.5).astype(int) *
                       idiv)

        size = N * h

        atoms.cell[i] *= size / np.linalg.norm(atoms.cell[i])

        # shift structure to the center
        shift_c[i] = (size - extension) / 2
        shift_c[i] -= lowest_c[i]

    atoms.translate(shift_c)


def adjust_cell_and_get_h(atoms: Atoms, border: float,
                          h: float = 0.2, idiv: int = 4):

    n_pbc = atoms.pbc.sum()

    if n_pbc != 3:
        adjust_cell(atoms, border, h, idiv)

    if n_pbc:
        N_c = h2gpts(h, atoms.cell, idiv)
        # logic from gpaw/utilitis/__init__.py
        h_c = (np.linalg.inv(atoms.cell)**2).sum(0)**-0.5 / N_c
        print(h_c)
        h = 0
        for pbc, h1 in zip(atoms.pbc, h_c):
            if pbc:
                h += h1 / n_pbc

        return h
    else:
        return h
