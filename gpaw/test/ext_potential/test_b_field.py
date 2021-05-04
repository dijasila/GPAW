from gpaw import GPAW
from gpaw.external import BField
from ase import Atoms
# import pytest


def test_b_field():
    L = 2.0
    atom = Atoms('H', magmoms=[1], cell=[L, L, L], pbc=True)
    atom.calc = GPAW(mode='pw',
                     external=BField(0.1))
    _ = atom.get_potential_energy()
