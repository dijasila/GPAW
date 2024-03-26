from ase.spacegroup import crystal
from gpaw import GPAW
from gpaw import PW
import pytest


def test_symmetry_fractional_translations_grid():
    'cristobalite'
    # no. 92 - tetragonal

    a = 5.0833674
    c = 7.0984738
    p0 = (0.2939118, 0.2939118, 0.0)
    p1 = (0.2412656, 0.0931314, 0.1739217)

    atoms = crystal(['Si', 'O'], basis=[p0, p1],
                    spacegroup=92, cellpar=[a, a, c, 90, 90, 90])

    with pytest.raises(ValueError, match=r"^The specified number"):
        failcalc = GPAW(mode=PW(),
                        xc='LDA',
                        kpts=(3, 3, 2),
                        nbands=40,
                        symmetry={'symmorphic': False},
                        gpts=(23, 23, 32),
                        eigensolver='rmm-diis')

        atoms.calc = failcalc
        atoms.get_potential_energy()
