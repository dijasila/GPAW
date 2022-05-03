"""Make sure we get an exception when an atom is too close to the boundary."""
import os
from ase import Atoms
from gpaw import GPAW
from gpaw.grid_descriptor import GridBoundsError
from gpaw.utilities import AtomsTooClose
import pytest


def test_too_close():
    a = 4.0
    x = 0.1
    hydrogen = Atoms('H', [(x, x, x)],
                     cell=(a, a, a))
    hydrogen.calc = GPAW(mode='fd')
    with pytest.raises((GridBoundsError, AtomsTooClose)):
        hydrogen.get_potential_energy()


@pytest.mark.skipif(not os.environ.get('GPAW_NEW'),
                    reason='ignore old code')
def test_too_close_to_boundary():
    a = 4.0
    x = 0.1
    hydrogen = Atoms('H', [(x, x, x)],
                     cell=(a, a, a),
                     pbc=(1, 1, 0))
    hydrogen.calc = GPAW()
    with pytest.raises(AtomsTooClose):
        hydrogen.get_potential_energy()
