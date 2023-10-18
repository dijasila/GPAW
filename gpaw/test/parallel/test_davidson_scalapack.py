import numpy as np
import pytest

from ase.build import bulk
from ase.parallel import world
from gpaw import GPAW, FermiDirac, PW


def get_calculator(sl_auto, kpoint_gamma, use_elpa=False):
    calculator = GPAW(
        mode=PW(100),
        nbands='400%',
        basis='szp(dzp)',
        txt=None,
        occupations=FermiDirac(0.1),
        parallel={'sl_auto': sl_auto, 'band': 2 if world.size == 8 else 1,
                  'use_elpa': use_elpa},
        kpts=[1, 1, 1] if kpoint_gamma else [1, 1, 2],
        symmetry='off',
        convergence={'maximum iterations': 1})
    return calculator


@pytest.mark.parametrize('kpoint_gamma', [True, False])
def test_davidson_scalapack_eigenvalues(scalapack, kpoint_gamma):
    atoms = bulk('Si', cubic=True) * [2, 1, 1]

    atoms1 = atoms.copy()
    atoms2 = atoms.copy()
    atoms3 = atoms.copy()

    atoms1.calc = get_calculator(sl_auto=True, kpoint_gamma=kpoint_gamma)
    atoms2.calc = get_calculator(sl_auto=False, kpoint_gamma=kpoint_gamma)
    atoms3.calc = get_calculator(sl_auto=True, kpoint_gamma=kpoint_gamma,
                                 use_elpa=True)

    assert_energies_equal(atoms1, atoms2)
    assert_energies_equal(atoms2, atoms3)


def assert_energies_equal(atoms1, atoms2):
    e1 = atoms1.get_potential_energy()
    e2 = atoms2.get_potential_energy()

    for n, kpoint in enumerate(atoms1.calc.get_bz_k_points()):
        eigvals1 = atoms1.calc.get_eigenvalues(kpt=n, spin=0)
        eigvals2 = atoms2.calc.get_eigenvalues(kpt=n, spin=0)

        assert np.allclose(eigvals1, eigvals2, rtol=1e-11)

    assert e1 == pytest.approx(e2, rel=1e-11)
