import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.etdm import ETDM
from gpaw.directmin.tools import excite
from gpaw.directmin.derivatives import Davidson
from gpaw.mom import prepare_mom_calculation
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
    H2O.center(vacuum=4.0)

    calc = GPAW(mode=LCAO(force_complex_dtype=True),
                h=0.22,
                occupations={'name': 'fixed-uniform'},
                eigensolver=ETDM(localizationtype='PM_PZ',
                                 localizationseed=42,
                                 functional_settings={
                                     'name': 'PZ-SIC',
                                     'scaling_factor': \
                                         (0.5, 0.5)}),  # SIC/2
                convergence={'eigenstates': 1e-4},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                spinpol=True,
                symmetry='off'
                )
    H2O.calc = calc
    H2O.get_potential_energy()

    f_sn = excite(calc, 0, 0, spin=(0, 0))
    prepare_mom_calculation(calc, H2O, f_sn)

    dave = Davidson(calc.wfs.eigensolver, None)
    appr_sp_order = dave.estimate_sp_order(calc)

    numeric = False
    if numeric:
        from ase.calculators.test import numeric_force
        f_num = np.array([[numeric_force(H2O, a, i)
                          for i in range(3)]
                         for a in range(len(H2O))])
        print('Numerical forces')
        print(f_num)
        print(f - f_num, np.abs(f - f_num).max())


    assert f2 == pytest.approx(f3, abs=0.1)

test_lcaosic_h2o(0)