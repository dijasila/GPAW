import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.etdm import ETDM
from ase import Atoms
import numpy as np


def test_lcaosic_h2o(in_tmp_dir):
    """
    test Perdew-Zunger Self-Interaction
    Correction  in LCAO mode using DirectMin
    :param in_tmp_dir:
    :return:
    """

    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * 104.51
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=5.0)

    calc = GPAW(mode=LCAO(force_complex_dtype=True),
                h=0.25,
                occupations={'name': 'fixed-uniform'},
                eigensolver={'name': 'etdm',
                             'localizationtype': 'PM',
                             'localizationseed': 42,
                             'functional_settings': {'name': 'PZ-SIC',
                                                     'scaling_factor': \
                                                         (0.5, 0.5)}},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off'
                )
    H2O.calc = calc
    H2O.get_potential_energy()

    homo = 3
    lumo = 4
    a = 0.5 * np.pi
    c = calc.wfs.kpt_u[0].C_nM.copy()
    calc.wfs.kpt_u[0].C_nM[homo] = np.cos(a) * c[homo] + np.sin(a) * c[lumo]
    calc.wfs.kpt_u[0].C_nM[lumo] = np.cos(a) * c[lumo] - np.sin(a) * c[homo]

    calc.set(eigensolver={'name': 'etdm',
                          'functional_settings': {'name': 'PZ-SIC',
                                                  'scaling_factor': \
                                                      (0.5, 0.5)},
                          'constraints': [[[homo], [lumo]]],
                          'need_init_orbs': False})

    e = H2O.get_potential_energy()

    assert e == pytest.approx(24.87379, abs=1.0e-4)
