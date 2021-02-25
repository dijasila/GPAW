import pytest
import numpy as np
from ase.build import bulk
from ase.parallel import world
from gpaw.utilities import compiled_with_sl
from gpaw import GPAW, FermiDirac, PW, Davidson

pytestmark = pytest.mark.skipif(
    world.size != 4 or not compiled_with_sl(),
    reason='world.size != 4 or not compiled_with_sl()')


def test_davidson_scalapack(in_tmp_dir):
    atoms = bulk('Ag', cubic=True) # Need cubic so there are enough electrons...
    atoms1 = atoms.copy()
    atoms2 = atoms.copy()
    calc_noscalapack = GPAW(mode=PW(340),
                    kpts=[1] * 3,
                    occupations=FermiDirac(0.1),
                    eigensolver=Davidson(use_scalapack=False))

    calc_scalapack = GPAW(mode=PW(340),
                    kpts=[1] * 3,
                    occupations=FermiDirac(0.1),
                    eigensolver=Davidson(use_scalapack=True))
                    
    atoms1.calc = calc_noscalapack
    atoms2.calc = calc_scalapack

    e1 = atoms1.get_potential_energy()
    e2 = atoms2.get_potential_energy()

    assert np.isclose(e1, e2, rtol=1e-5)