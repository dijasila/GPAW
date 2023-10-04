import pytest

from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import size


def test_noncollinear_o2(in_tmp_dir, gpaw_new):
    if size > 2:
        pytest.skip('mpi world size >2')
    if not gpaw_new:
        pytest.skip('fatal crash with old code')

    a = Atoms('OO', [[0, 0, 0], [0, 0, 1.1]], magmoms=[1, 1], pbc=(1, 0, 0))
    a.center(vacuum=2.5)
    a.calc = GPAW(mode='pw',
                  kpts=(2, 1, 1))
    f0 = a.get_forces()
    e0 = a.calc.get_eigenvalues(0, 0)[5]
    p0 = a.calc.get_pseudo_wave_function(5, periodic=True)

    a.calc = GPAW(mode='pw',
                  kpts=(2, 1, 1),
                  symmetry='off',
                  experimental={'magmoms': [[0, 0.5, 0.5], [0, 0, 1]]})
    f = a.get_forces()
    e = a.calc.get_eigenvalues(0, 0)[10]
    p = a.calc.get_pseudo_wave_function(10, periodic=True)

    assert abs(f - f0).max() < 0.01
    assert e == pytest.approx(e0, abs=0.002)
    assert abs(p0)**2 == pytest.approx((abs(p)**2).sum(axis=0), abs=1e-4)

    a.calc.write('o2.gpw')
    a.calc.write('o2w.gpw', 'all')
    calc = GPAW('o2w.gpw')

    p = calc.get_pseudo_wave_function(10, periodic=True)
    assert abs(p0)**2 == pytest.approx((abs(p)**2).sum(axis=0), abs=1e-4)

    if gpaw_new:
        n_sR = calc.calculation.densities().all_electron_densities()
        print(n_sR)
