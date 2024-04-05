from math import sqrt
import pytest

from ase import Atoms


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
