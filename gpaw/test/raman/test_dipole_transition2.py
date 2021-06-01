import numpy as np
import pytest

from ase.parallel import world, parprint
from ase.units import Bohr
from gpaw import GPAW
from gpaw.raman.dipoletransition import get_dipole_transitions
from gpaw.lrtddft.kssingle import KSSingles


def test_dipole_transition(gpw_files, tmp_path_factory):
    """Check dipole matrix-elements for Li."""
    calc = GPAW(gpw_files['bcc_li_lcao_wfs'])
    dip_skvnm = get_dipole_transitions(calc.atoms, calc, savetofile=False,
                                       realdipole=True)
    parprint("Dipole moments calculated")
    assert dip_skvnm.shape == (1, 4, 3, 4, 4)
    dip_kvnm = dip_skvnm[0] * Bohr

    print(world.rank, dip_kvnm[:, 0, 0, 3])

    # Check numerical value of a few elements - signs might change!
    assert 0.0823 == pytest.approx(abs(dip_kvnm[0, 0, 0, 1]), abs=1e-4)

    calc = GPAW(gpw_files['bcc_li_fd_wfs'])
    # compare to lrtddft implementation
    kss = KSSingles()
    atoms = calc.atoms
    atoms.calc = calc
    kss.calculate(calc.atoms, 1)
    lrref = []
    lrrefv = []
    for ex in kss:
        lrref.append(-1. * ex.mur * Bohr)
        lrrefv.append(-1. * ex.muv * Bohr)
    lrref = np.array(lrref)
    lrrefv = np.array(lrrefv)

    # some printout for manual inspection, if wanted
    parprint("                   r-gauge          lrtddft(v)         raman(v)")
    f = "{} {:+.4f}    {:+.4f}    {:+.4f}"
    parprint(f.format('k=0, 0->1 (x)',
                      lrref[0, 0], lrrefv[0, 0], dip_kvnm[0, 0, 0, 1]))
    parprint(f.format('k=0, 0->1 (y)',
                      lrref[0, 1], lrrefv[0, 1], dip_kvnm[0, 1, 0, 1]))
    parprint(f.format('k=0, 0->1 (z)',
                      lrref[0, 2], lrrefv[0, 2], dip_kvnm[0, 2, 0, 1]))

    parprint(f.format('k=0, 0->2 (x)',
                      lrref[1, 0], lrrefv[1, 0], dip_kvnm[0, 0, 0, 2]))
    parprint(f.format('k=0, 0->2 (y)',
                      lrref[1, 1], lrrefv[1, 1], dip_kvnm[0, 1, 0, 2]))
    parprint(f.format('k=0, 0->2 (z)',
                      lrref[1, 2], lrrefv[1, 2], dip_kvnm[0, 2, 0, 2]))

    parprint(f.format('k=0, 0->3 (x)',
                      lrref[2, 0], lrrefv[2, 0], dip_kvnm[0, 0, 0, 3]))
    parprint(f.format('k=0, 0->3 (y)',
                      lrref[2, 1], lrrefv[2, 1], dip_kvnm[0, 1, 0, 3]))
    parprint(f.format('k=0, 0->3 (z)',
                      lrref[2, 2], lrrefv[2, 2], dip_kvnm[0, 2, 0, 3]))
    parprint("")
    parprint(f.format('k=1, 0->1 (x)',
                      lrref[3, 0], lrrefv[3, 0], dip_kvnm[1, 0, 0, 1]))
    parprint(f.format('k=1, 0->1 (y)',
                      lrref[3, 1], lrrefv[3, 1], dip_kvnm[1, 1, 0, 1]))
    parprint(f.format('k=1, 0->1 (z)',
                      lrref[3, 2], lrrefv[3, 2], dip_kvnm[1, 2, 0, 1]))

    parprint(f.format('k=1, 0->2 (x)',
                      lrref[4, 0], lrrefv[4, 0], dip_kvnm[1, 0, 0, 2]))
    parprint(f.format('k=1, 0->2 (y)',
                      lrref[4, 1], lrrefv[4, 1], dip_kvnm[1, 1, 0, 2]))
    parprint(f.format('k=1, 0->2 (z)',
                      lrref[4, 2], lrrefv[4, 2], dip_kvnm[1, 2, 0, 2]))

    parprint(f.format('k=1, 0->3 (x)',
                      lrref[5, 0], lrrefv[5, 0], dip_kvnm[1, 0, 0, 3]))
    parprint(f.format('k=1, 0->3 (y)',
                      lrref[5, 1], lrrefv[5, 1], dip_kvnm[1, 1, 0, 3]))
    parprint(f.format('k=1, 0->3 (z)',
                      lrref[5, 2], lrrefv[5, 2], dip_kvnm[1, 2, 0, 3]))
