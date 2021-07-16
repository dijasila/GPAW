import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.lcao.tools import excite
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO

from ase import Atoms
import numpy as np


def test_mom_directopt_lcao(in_tmp_dir):
    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * 104.51
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=5.0)

    calc = GPAW(mode=LCAO(),
                basis='dzp',
                h=0.20,
                occupations={'name': 'fixed-uniform'},
                eigensolver='direct-min-lcao',
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                spinpol=True
                )
    H2O.calc = calc
    H2O.get_potential_energy()

    calc.set(eigensolver=DirectMinLCAO(searchdir_algo={'name': 'LSR1P',
                                                       'method': 'LSR1'},
                                       linesearch_algo={'name': 'UnitStep'},
                                       need_init_orbs=False))
    # Ground-state occupation numbers
    f_sn = []
    for s in range(2):
        f_sn.append(calc.get_occupation_numbers(spin=s))
    excite(f_sn, 0, 0, spin=(0, 0))
    prepare_mom_calculation(calc, H2O, f_sn)

    def rotate_homo_lumo(calc=calc):
        a = np.pi / 2
        iters = calc.get_number_of_iterations()
        if iters == 5:
            c = calc.wfs.kpt_u[0].C_nM.copy()
            calc.wfs.kpt_u[0].C_nM[3] = np.cos(a) * c[3] + np.sin(a) * c[4]
            calc.wfs.kpt_u[0].C_nM[4] = np.cos(a) * c[4] - np.sin(a) * c[3]

    calc.attach(rotate_homo_lumo, 1)

    e = H2O.get_potential_energy()

    assert e == pytest.approx(-5.091912426348663, abs=1.0e-4)

