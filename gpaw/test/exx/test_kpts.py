"""Test case where q=k1-k2 has component outside 0<=q<1 range."""
from typing import Tuple

import pytest
import numpy as np
from ase import Atoms

from gpaw import GPAW, PW
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.xc.exx import EXX
from gpaw.mpi import world

n = 7


@pytest.fixture(scope='module')
def atoms():
    a = Atoms('HH',
              cell=[2, 2, 2.5, 90, 90, 60],
              pbc=1,
              positions=[[0, 0, 0], [0, 0, 0.75]])
    a.calc = GPAW(mode=PW(200),
                  kpts=(n, n, 1),
                  xc='PBE')
    a.get_potential_energy()
    return a


def bandgap(eps: np.ndarray) -> Tuple[int, int, float]:
    """Find band-gap."""
    k1 = eps[0, :, 0].argmax()
    k2 = eps[0, :, 1].argmin()
    return k1, k2, eps[0, k2, 1] - eps[0, k1, 0]


gaps = {'EXX': 21.45,
        'PBE0': 13.93,
        'HSE06': 14.44,
        'PBE': 11.63}


@pytest.mark.libxc
@pytest.mark.parametrize('xc', ['EXX', 'PBE0', 'HSE06'])
def test_kpts(xc, atoms):
    c = atoms.calc
    e0, v0, v = non_self_consistent_eigenvalues(c, xc)
    e = e0 - v0 + v
    k1, k2, gap = bandgap(e)
    assert k1 == 4 and k2 == 7
    assert gap == pytest.approx(gaps[xc], abs=0.01)
    k1, k2, gap = bandgap(e0)
    assert k1 == 4 and k2 == 7
    assert gap == pytest.approx(gaps['PBE'], abs=0.01)
    if world.size == 1:
        xc2 = EXX(c, xc=xc, bands=(0, c.wfs.bd.nbands), txt=None)
        xc2.calculate()
        v2 = xc2.get_eigenvalue_contributions()
        assert np.allclose(v, v2), (v, v2)
