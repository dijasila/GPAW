import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW


def test_direct_optimization():
    a = 2.0
    atoms = Atoms('Li',
                  cell=[a, a, a],
                  pbc=True)
    atoms.calc = GPAW(
        mode='pw',
        kpts=(2, 2, 2),
        nbands=-2,
        mixer={'backend': 'no-mixing'},
        occupations={"name": "fixed-uniform"},
        eigensolver={"name": "lbfgs", "memory": 2},
        convergence={"bands": "occupied"},
    )
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(0.480154, rel=0.05)
