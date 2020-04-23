"""Test case where q=k1-k2 has component outside 0<=q<1 range."""

import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.xc.exx import EXX
from gpaw.mpi import world

pytestmark = pytest.mark.skipif(world.size > 1, reason='Not parallelized')

n = 7


@pytest.fixture
def atoms():
    a = Atoms('HH',
              cell=[2, 2, 2.5, 90, 90, 60],
              pbc=1,
              positions=[[0, 0, 0], [0, 0, 0.75]])
    a.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups='ae',
                  kpts=(n, n, 1),
                  xc='PBE')
    a.get_potential_energy()
    return a


@pytest.mark.parametrize('xc', ['EXX', 'PBE0', 'HSE06'])
def test_kpts(xc, atoms):
    c = atoms.calc
    eps = non_self_consistent_eigenvalues(c, xc)[2]
    xc2 = EXX(c, xc=xc, bands=(0, c.wfs.bd.nbands), txt=None)
    xc2.calculate()
    eps0 = xc2.get_eigenvalue_contributions()
    assert np.allclose(eps0, eps), (eps0, eps)
