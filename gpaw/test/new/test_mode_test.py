import pytest
from ase import Atoms
from ase.calculators.test import numeric_forces, numeric_stress
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from gpaw.new.ase_interface import GPAW


def test_fake_mode():
    atoms = Atoms('LiH',
                  [[0, 0.1, 0.2],
                   [0, 0, 1.4]])
    atoms.calc = GPAW(
        mode='fake',
        symmetry='off')
    f1 = atoms.get_forces()
    f2 = numeric_forces(atoms)
    assert abs(f1 - f2).max() < 0.0005


def test_fake_mode_bulk():
    a = 2.0
    atoms = Atoms('Li',
                  cell=[a, a, a],
                  pbc=True)
    atoms.calc = GPAW(
        mode='fake',
        kpts=(2, 2, 2))
    f = atoms.get_forces()
    assert abs(f).max() < 0.0001
    e = atoms.get_potential_energy()
    s = atoms.get_stress()
    print(a, e, s)
    s2 = numeric_stress(atoms)
    print(s2)
    assert abs(s - s2).max() < 0.0001
    BFGS(ExpCellFilter(atoms)).run(fmax=0.002)
    s = atoms.get_stress()
    print(s)
    assert abs(s).max() < 0.0001
    assert atoms.cell[0, 0] == pytest.approx(2.044, abs=0.0005)
