import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW

a = 2.5
k = 4


def test_afm_h_chain():
    """Compare 2*H AFM cell with 1*H q=1/2 spin-spiral cell."""
    h = Atoms('H',
              magmoms=[1],
              cell=[a, 0, 0],
              pbc=[1, 0, 0])
    h.center(vacuum=2.0, axis=(1, 2))
    h.calc = GPAW(mode={'name': 'pw',
                        'ecut': 400,
                        'qspiral': [0.5, 0, 0]},
                  magmoms=[[1, 0, 0]],
                  symmetry='off',
                  kpts=(2 * k, 1, 1))
    e1 = h.get_potential_energy()
    h1, l1 = h.calc.get_homo_lumo()

    h2 = Atoms('H2',
               [(0, 0, 0), (a, 0, 0)],
               magmoms=[1, -1],
               cell=[2 * a, 0, 0],
               pbc=[1, 0, 0])
    h2.center(vacuum=2.0, axis=(1, 2))
    h2.calc = GPAW(mode={'name': 'pw',
                         'ecut': 400},
                   kpts=(k, 1, 1))
    e2 = h2.get_potential_energy()
    h2, l2 = h2.calc.get_homo_lumo()

    assert 2 * e1 == pytest.approx(e2, abs=0.002)
    assert h1 == pytest.approx(h2, abs=0.001)
    assert l1 == pytest.approx(l2, abs=0.001)
