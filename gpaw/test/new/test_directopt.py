import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW


def test_direct_optimization():
    """test with 4 cores to check parallelization over
     both kpts and domain"""

    a = 2.0
    atoms = Atoms('Li',
                  cell=[a, a, a],
                  pbc=True)
    atoms.calc = GPAW(
        mode='pw',
        kpts=(1, 1, 2),
        nbands=-2,
        symmetry="off",
        mixer={'backend': 'no-mixing'},
        occupations={"name": "fixed-uniform"},
        eigensolver={"name": "lbfgs", "memory": 2},
        convergence={"bands": "occupied"},
    )
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(-2.3329, rel=0.05)
