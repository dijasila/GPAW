import pytest

import numpy as np
from ase.units import Ha

from gpaw.hubbard import hubbard


def test_hubbard_U_noncollinear():
    rng = np.random.default_rng(seed=555)

    # Generate random data simulating a setup with
    # s and p bounded and unbounded partial waves.

    D_sii = (rng.random([4, 8, 8], dtype=np.float64)
             + 1j * rng.random([4, 8, 8], dtype=np.float64))
    # Make Hermitian
    D_sii = (D_sii + np.transpose(D_sii.conj(), (0, 2, 1))) / 2
    l_j = [0, 1, 0, 1]

    # Hubbard correction on the p orbitals
    l = 1
    U = 4  # eV

    # Generate inner products for the p-orbitals
    lq = np.zeros(len(l_j) * (len(l_j) + 1) // 2)
    lq[4] = 0.9  # Bounded-bounded
    lq[6] = 1.1  # Bounded-unbounded
    lq[9] = 1.2  # Unbounded-unbounded

    eU, _ = hubbard(D_sii, U / Ha, l, l_j, lq, scale=True)
    eU_onlyreal, _ = hubbard(D_sii.real, U / Ha, l, l_j, lq, scale=True)

    # Assert that disregarding the complex values will alter the energy
    assert eU == pytest.approx(-7.825104, abs=1.e-2)
    assert eU_onlyreal == pytest.approx(-7.546553, abs=1.e-2)
