import numpy as np
import pytest
from gpaw.zero_field_splitting.paw import coulomb, coulomb_integrals


def test_paw_coulomb_correction():
    from gpaw.setup import create_setup
    from gpaw.utilities import packed_index
    setup = create_setup('H', lmax=2)
    things = coulomb(setup.rgd,
                     np.array(setup.data.phi_jg),
                     np.array(setup.data.phit_jg),
                     setup.l_j,
                     setup.g_lg)
    C_iiii = coulomb_integrals(setup.rgd, setup.l_j, *things)
    ni = len(C_iiii)
    for i1 in range(ni):
        for i2 in range(ni):
            p12 = packed_index(i1, i2, ni)
            for i3 in range(ni):
                for i4 in range(ni):
                    p34 = packed_index(i3, i4, ni)
                    assert setup.M_pp[p12, p34] == pytest.approx(
                        C_iiii[i1, i2, i3, i4], abs=1e-5)
