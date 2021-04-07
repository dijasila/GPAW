import numpy as np
import pytest

from gpaw import GPAW
from gpaw.raman.dipoletransition import get_dipole_transitions
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.kssingle import KSSingles

def test_dipole_transition(gpw_files, tmp_path_factory):
    """Check dipole matrix-elements for bcc Li."""
    calc = GPAW(gpw_files['h2o_lcao_wfs'])
    dip_svknm = get_dipole_transitions(calc.atoms, calc, savetofile=False).real
    assert dip_svknm.shape == (1, 3, 1, 6, 6)

    for i in range(3):
        # d2 = np.real(dip_svknm[0, i, 0] * dip_svknm[0, i, 0].conj())
        assert(np.allclose(dip_svknm[0, i, 0] + dip_svknm[0, i, 0].T, 0.,
                           atol=1e-4))
        # assert(np.allclose(d2, d2.T, rtol=1e-3, atol=1e-5))

    # Check numerical value of a few elements
    assert -0.3265 == pytest.approx(dip_svknm[0, 0, 0, 0, 3], abs=1e-4)
    assert  0.1411 == pytest.approx(dip_svknm[0, 0, 0, 2, 3], abs=1e-4)
    assert -0.0987 == pytest.approx(dip_svknm[0, 0, 0, 3, 4], abs=1e-4)
    assert +0.3265 == pytest.approx(dip_svknm[0, 0, 0, 3, 0], abs=1e-4)
    assert +0.3889 == pytest.approx(dip_svknm[0, 1, 0, 0, 1], abs=1e-4)
    assert -0.3669 == pytest.approx(dip_svknm[0, 2, 0, 0, 2], abs=1e-4)

    for c in range(3):
        d = np.real(dip_svknm[0, c, 0])
        for i in range(6):
            f = "{:+.4f} {:+.4f} {:+.4f} {:+.4f} {:+.4f} {:+.4f}"
            print(f.format(d[i, 0], d[i, 1], d[i, 2], d[i, 3], d[i, 4],
                           d[i, 5]))
        print("")

    # compare to utilities implementation
    ref = dipole_matrix_elements_from_calc(calc, 0, 6)[0]
    assert(ref.shape == (6, 6, 3))
    # NOTE: So this might be a bit of a problem.
    # This implementation and the utilities implementation do not give the same
    # results. Similar, but not identical.
    # print(ref[:,:,0])

    # compare to lrtddft implementation
    # lr = LrTDDFT(calc)
    kss = KSSingles()
    atoms = calc.atoms
    atoms.calc = calc
    kss.calculate(calc.atoms, 1)
    ref2 = []
    for ex in kss:
        ref2.append(ex.mur)
    ref2 = np.array(ref2)
    # print(kss)
    print("        lrtddft(r) utilities(r) raman(v)")
    f = "{} {:+.4f}    {:+.4f}    {:+.4f}"
    print(f.format('0->4 (z)', ref2[0, 2], ref[0, 4, 2],
                   dip_svknm[0, 2, 0, 0, 4]))
    print(f.format('0->5 (y)', ref2[1, 1], ref[0, 5, 1],
                   dip_svknm[0, 1, 0, 0, 5]))
    print(f.format('1->4 (y)', ref2[2, 1], ref[1, 4, 1],
                   dip_svknm[0, 1, 0, 1, 4]))
    print(f.format('1->5 (z)', ref2[3, 2], ref[1, 5, 2],
                   dip_svknm[0, 2, 0, 1, 5]))
    print(f.format('2->4 (z)', ref2[4, 2], ref[2, 4, 2],
                   dip_svknm[0, 2, 0, 2, 4]))
    print(f.format('2->5 (y)', ref2[5, 1], ref[2, 5, 1],
                   dip_svknm[0, 1, 0, 2, 5]))
    print(f.format('3->4 (x)', ref2[6, 0], ref[3, 4, 0],
                   dip_svknm[0, 0, 0, 3, 4]))
