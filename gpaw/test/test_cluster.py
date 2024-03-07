from math import sqrt

from ase import Atoms
from ase.build import fcc111

from gpaw.cluster import Cluster, adjust_cell
from gpaw.mpi import world

from gpaw.utilities import h2gpts

import numpy as np

import pytest


def test_cluster():
    R = 2.0
    CO = Atoms('CO', [(1, 0, 0), (1, 0, R)])

    # I/O
    fxyz = 'CO.xyz'
    fpdb = 'CO.pdb'

    cell = [2., 3., R + 2.]
    CO.set_cell(cell, scale_atoms=True)
    world.barrier()
    CO.write(fxyz)
    world.barrier()
    CO_b = Cluster(filename=fxyz)
    assert len(CO) == len(CO_b)
    offdiagonal = CO_b.get_cell().sum() - CO_b.get_cell().diagonal().sum()
    assert offdiagonal == 0.0

    world.barrier()
    CO.write(fpdb)

    # read xyz files with additional info
    read_with_additional = True
    if read_with_additional:
        if world.rank == 0:
            f = open(fxyz, 'w')
            print("""2
    C 0 0 0. 1 2 3
    O 0 0 1. 6. 7. 8.""", file=f)
            f.close()

        world.barrier()

        CO = Cluster(filename=fxyz)


def test_CO():
    R = 2.0
    CO = Atoms('CO', [(1, 0, 0), (1, 0, R)])

    CO.rotate(90, 'y')
    assert CO.positions[1, 0] == pytest.approx(R, abs=1e-10)

    # translate
    CO.translate(-CO.get_center_of_mass())
    p = CO.positions.copy()
    for i in range(2):
        assert p[i, 1] == pytest.approx(0, abs=1e-10)
        assert p[i, 2] == pytest.approx(0, abs=1e-10)

    CO.rotate(p[1] - p[0], (1, 1, 1))
    q = CO.positions.copy()
    for c in range(3):
        assert q[0, c] == pytest.approx(p[0, 0] / sqrt(3), abs=1e-10)
        assert q[1, c] == pytest.approx(p[1, 0] / sqrt(3), abs=1e-10)


def test_non_periodic():
    R = 2.0
    b = 4.0
    h = 0.2

    CO = Atoms(['C', 'O'], [(1, 0, 0), (1, 0, R)])

    adjust_cell(CO, b, h)
    cc = CO.get_cell()

    for c in range(3):
        width = 2 * b
        if c == 2:
            width += R + 2 * h
        assert cc[c, c] == pytest.approx(width, abs=1e-10)


def test_non_orthogonal_unitcell():
    a = 3.912
    vac = 2
    box = 3.
    h = 0.2

    atoms = (fcc111('Pt', (1, 1, 1), a=a, vacuum=vac))

    atoms.pbc = [1, 1, 0]

    cell0 = atoms.cell.copy()

    adjust_cell(atoms, box, h)

    # check that the box ajusts for h in only non periodic directions
    assert atoms.cell[:1, :1] == pytest.approx(cell0[:1, :1])
    # check that the atom is shifted in non periodic direction
    assert atoms.positions[0, 2] >= vac

    h_c = np.zeros(3)
    N_c = h2gpts(h, atoms.cell)
    for i in range(3):
        h_c[i] = np.linalg.norm(atoms.cell / N_c)

    assert h_c[2] == pytest.approx(h_c[:2].sum() / 2)


def test_platinum_surface():
    """ensure that non-periodic direction gets modified only"""
    surface = fcc111('Pt', (5, 6, 2), a=3.912, orthogonal=True, vacuum=2)
    original_cell = surface.cell.copy()

    h = 0.2
    vacuum = 4
    adjust_cell(surface, vacuum, h=h)

    # perdiodic part shall not be changed
    assert (original_cell[:2] == surface.cell[:2]).all()
    # the surfcae is shifted upwards
    assert (surface.get_positions().T[2] >= vacuum).all()
    # what internally is done in gpaw
    N_c = h2gpts(h, surface.cell)
    h_c = np.diag(surface.cell / N_c)

    h_z = h_c[:2].sum() / 2  # average of x and y
    assert h_z == pytest.approx(h_c[2])

def test_pbc_uncompleat_initial_unitcell():
    """ensure that non-periodic direction gets modified only"""
    surface = fcc111('Pt', (5, 6, 2), a=3.912, orthogonal=True)
    original_cell = surface.cell.copy()

    h = 0.2
    vacuum = 4
    adjust_cell(surface, vacuum, h=h)

    # perdiodic part shall not be changed
    assert (original_cell[:2] == surface.cell[:2]).all()
    # the surfcae is shifted upwards
    assert (surface.get_positions().T[2] >= vacuum).all()
    
    # what internally is done in gpaw
    N_c = h2gpts(h, surface.cell)
    h_c = np.diag(surface.cell / N_c)

    h_z = h_c[:2].sum() / 2  # average of x and y
    assert h_z == pytest.approx(h_c[2])
