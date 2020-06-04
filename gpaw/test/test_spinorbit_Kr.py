import pytest
from gpaw.mpi import world
from ase import Atoms
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates
from gpaw.test import equal

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


def test_spinorbit_Kr():
    a = Atoms('Kr')
    a.center(vacuum=3.0)

    calc = GPAW(mode='pw', xc='LDA')

    a.calc = calc
    a.get_potential_energy()

    e_n = calc.get_eigenvalues()
    e_m = soc_eigenstates(calc)['e_km'][0]

    equal(e_n[0] - e_m[0], 0.0, 1.0e-3)
    equal(e_n[1] - e_m[2], 0.452, 1.0e-3)
    equal(e_n[2] - e_m[4], -0.226, 1.0e-3)
