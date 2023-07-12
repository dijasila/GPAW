import pytest

from gpaw import GPAW, PW
from ase import Atoms
import numpy as np
from gpaw.directmin.etdm_fdpw import FDPWETDM


def test_pz_localization_pw(in_tmp_dir):
    """
    Test Perdew-Zunger and Kohn-Sham localizations in PW mode
    :param in_tmp_dir:
    :return:
    """

    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * (104.51 + 2.0)
    eps = 0.02
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d + eps, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=3.0)

    calc = GPAW(mode=PW(300, force_complex_dtype=True),
                occupations={'name': 'fixed-uniform'},
                convergence={'energy': np.inf,
                             'eigenstates': np.inf,
                             'density': np.inf,
                             'minimum iterations': 0},
                eigensolver=FDPWETDM(
                    functional_settings={'name': 'PZ-SIC',
                                         'scaling_factor': (0.5, 0.5)  # SIC/2
                                         },
                    localizationseed=42,
                    localizationtype='KS_PZ',
                    localization_tol=5e-2,
                    g_tol=5.0e-2,
                    converge_unocc=False),
                mixer={'backend': 'no-mixing'},
                symmetry='off',
                spinpol=True
                )
    H2O.calc = calc
    e = H2O.get_potential_energy()
    assert e == pytest.approx(-10.1, abs=0.1)
