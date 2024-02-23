from math import sqrt

from ase import Atoms
from ase.build import fcc111

from gpaw.cluster import Cluster
from gpaw.mpi import world
from gpaw.core import UGDesc

import pytest


def test_cluster():
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

    # rotate the nuclear axis to the direction (1,1,1)
    CO.rotate(p[1] - p[0], (1, 1, 1))
    q = CO.positions.copy()
    for c in range(3):
        assert q[0, c] == pytest.approx(p[0, 0] / sqrt(3), abs=1e-10)
        assert q[1, c] == pytest.approx(p[1, 0] / sqrt(3), abs=1e-10)

    # minimal box
    b = 4.0
    CO = Cluster(['C', 'O'], [(1, 0, 0), (1, 0, R)])
    CO.minimal_box(b)
    cc = CO.get_cell()
    for c in range(3):
        width = 2 * b
        if c == 2:
            width += R
        assert cc[c, c] == pytest.approx(width, abs=1e-10)

    # minimal box, ensure multiple of 4
    h = .13
    b = [2, 3, 4]
    CO.minimal_box(b, h=h)
    cc = CO.get_cell()
    for c in range(3):
        # print "cc[c,c], cc[c,c] / h % 4 =", cc[c, c], cc[c, c] / h % 4
        for a in CO:
            print(a.symbol, b[c], a.position[c], cc[c, c] - a.position[c])
            assert a.position[c] > b[c]
        assert cc[c, c] / h % 4 == pytest.approx(0.0, abs=1e-10)

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


def test_minimal_box_mixed_pbc():
    # Orthogonal unit cell
    atoms = Cluster(Atoms('H'))
    atoms.center(vacuum=2.)
    atoms.pbc = [0, 1, 1]
    cell0 = atoms.cell.copy()

    box = 3
    atoms.minimal_box(box)

    # chack that the periodic part dont change
    assert atoms.cell[1:, 1:] == pytest.approx(cell0[1:, 1:])
    assert atoms.cell[0, 0] == 2 * box

    _, h = atoms.minimal_box(box, 0.2)

    # check that the box ajusts for h in only non periodic directions
    assert atoms.cell[1:, 1:] == pytest.approx(cell0[1:, 1:])
    assert atoms.cell[0, 0] / h % 4 == pytest.approx(0)

    grid = UGDesc.from_cell_and_grid_spacing(atoms.cell, h, atoms.pbc)
    h_c = grid._gd.get_grid_spacings()

    assert h_c[0] == pytest.approx(h_c[1:].sum() / 2)

    # check for non square unit cell
    atoms.cell[1, 1] = 3
    _, h = atoms.minimal_box(box, h=0.2)

    assert atoms.cell[0, 0] / h % 4 == pytest.approx(0)

    grid = UGDesc.from_cell_and_grid_spacing(atoms.cell, h, atoms.pbc)
    h_c = grid._gd.get_grid_spacings()

    assert h_c[0] == pytest.approx(h_c[1:].sum() / 2)

    # testing non orthogonal uint cell
    a = 3.912
    vac = 2
    atoms = Cluster(fcc111('Pt', (1, 1, 1), a=a, vacuum=vac))

    atoms.pbc = [1, 1, 0]

    cell0 = atoms.cell.copy()
    atoms.minimal_box(box)

    assert atoms.cell[2, 2] == 2 * box
    # chack that the periodic part don't change
    assert atoms.cell[:1, :1] == pytest.approx(cell0[:1, :1])

    _, h = atoms.minimal_box(box, h=0.2)
    # check that the box ajusts for h in only non periodic directions
    assert atoms.cell[:1, :1] == pytest.approx(cell0[:1, :1])
    assert atoms.cell[2, 2] / h % 4 == pytest.approx(0)

    grid = UGDesc.from_cell_and_grid_spacing(atoms.cell, h, atoms.pbc)
    h_c = grid._gd.get_grid_spacings()

    assert h_c[2] == pytest.approx(h_c[:2].sum() / 2)


def test_platinum_surface():
    """ensure that non-periodic direction gets modified only"""
    surface = fcc111('Pt', (5, 6, 2), a=3.912, orthogonal=True, vacuum=2)
    original_cell = surface.cell.copy()

    h = 0.2
    vacuum = 4
    multiple = 4

    surface = Cluster(surface)
    _, h = surface.minimal_box(vacuum, h=h, multiple=multiple)

    # perdiodic part shall not be changed
    assert (original_cell[:2] == surface.cell[:2]).all()
    # the surfcae is shifted upwards
    assert (surface.get_positions().T[2] >= vacuum).all()
    # what internally is done in gpaw
    grid = UGDesc.from_cell_and_grid_spacing(surface.cell, h, surface.pbc)
    h_c = grid._gd.get_grid_spacings()

    h_z = h_c[:2].sum() / 2  # average of x and y

    # This one works with the h calculated by the cluster and to see that
    # it is a multipe of 4
    assert surface.cell[2, 2] / h % multiple == pytest.approx(0)
    # the prblem is here the the new h_c[2] in z derecton dose not become the
    # same as the h we feed the grid calculation I DONT KNOW HOW TO FIX THIS
    assert h_c[2] == h
    assert h_z == pytest.approx(h_c[2])
