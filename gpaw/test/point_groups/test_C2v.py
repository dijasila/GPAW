import pytest
import numpy as np
from ase.build import molecule
from gpaw import GPAW
from gpaw.point_groups import SymmetryChecker, PointGroup


def test_c2v():
    atoms = molecule('H2O', cell=[8, 8, 8], pbc=1)
    atoms.center()
    atoms.calc = calc = GPAW(mode='lcao', txt='h2o.txt')
    atoms.get_potential_energy()
    calc.write('h2o.gpw', mode='all')
    C = atoms.positions[0]
    C = atoms.cell.sum(0) / 2
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
