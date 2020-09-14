import pytest
import numpy as np
from gpaw import GPAW
from gpaw.point_groups import SymmetryChecker, PointGroup


def test_c2v(gpw_files):
    calc = GPAW(gpw_files['h2o_lcao_wfs'])
    C = calc.atoms.positions[0]
    pg = PointGroup('C2v')
    sc = SymmetryChecker(pg, C, 2.0)
    symmetries = ''
    for n in range(4):
        print('-' * 70)
        norm, ovl, characters = sc.check_band(calc, n)
        index = np.argmax(characters)
        symmetries += pg.symmetries[index]
        assert characters[index] == pytest.approx(norm, abs=0.05)
    assert symmetries == 'A1B2A1B1'
