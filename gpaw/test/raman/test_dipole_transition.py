import numpy as np
import pytest

from gpaw import GPAW
from gpaw.raman.dipoletransition import get_dipole_transitions


def test_dipole_transition(gpw_files, tmp_path_factory):
    """Check dipole matrix-elements for bcc Li."""
    calc = GPAW(gpw_files['h2o_lcao_wfs'])
    dip_svknm = get_dipole_transitions(calc.atoms, calc, savetofile=False)

    assert dip_svknm.shape == (1, 3, 1, 6, 6)

    for i in range(3):
        # d2 = np.real(dip_svknm[0, i, 0] * dip_svknm[0, i, 0].conj())
        assert(np.allclose(dip_svknm[0, i, 0] + dip_svknm[0, i, 0].T, 0.,
                           atol=1e-4))
        # assert(np.allclose(d2, d2.T, rtol=1e-3, atol=1e-5))

    # Check numerical value of a few elements
    assert -0.3265 == pytest.approx(dip_svknm[0, 0, 0, 0, 3], abs=1e-4)
    assert +0.3265 == pytest.approx(dip_svknm[0, 0, 0, 3, 0], abs=1e-4)
    assert +0.3889 == pytest.approx(dip_svknm[0, 1, 0, 0, 1], abs=1e-4)
    assert +0.3669 == pytest.approx(dip_svknm[0, 2, 0, 0, 2], abs=1e-4)

    for c in range(3):
        d = np.real(dip_svknm[0, c, 0])
        for i in range(6):
            f = "{:+.4f} {:+.4f} {:+.4f} {:+.4f} {:+.4f} {:+.4f}"
            print(f.format(d[i, 0], d[i, 1], d[i, 2], d[i, 3], d[i, 4],
                           d[i, 5]))
        print("")
