from math import sqrt
import pytest

from ase import Atoms
from ase.build import fcc111

from gpaw.cluster import adjust_cell

from gpaw.utilities import h2gpts
from gpaw.grid_descriptor import GridDescriptor


def test_CO(in_tmp_dir):
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
    box = 3.
    h = 0.2

    for atoms in [
            fcc111('Pt', (1, 1, 1), a=a),
            fcc111('Pt', (5, 6, 2), a=3.912, orthogonal=True)]:
        old_cell = atoms.cell.copy()

        adjust_cell(atoms, box, h)

        # check that the box ajusts for h in only non periodic directions
        assert atoms.cell[:2, :2] == pytest.approx(old_cell[:2, :2])
        # check that the atom is shifted in non periodic direction
        assert (atoms.positions[:, 2] >= box).all()

        N_c = h2gpts(h, atoms.cell)
        gd = GridDescriptor(N_c, atoms.cell, atoms.pbc)
        h_c = gd.get_grid_spacings()

        assert h_c[2] == pytest.approx(h_c[:2].sum() / 2)
