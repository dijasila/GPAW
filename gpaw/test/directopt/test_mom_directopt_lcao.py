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
                spinpol=True,
                convergence={'energy': 1e-3,
                             'density': 1e-3,
                             'eigenstates': 1e-3})
    H2O.calc = calc
    H2O.get_potential_energy()

    calc.set(eigensolver=DirectMinLCAO(searchdir_algo={'name': 'LSR1P',
                                                       'method': 'LSR1'},
                                       linesearch_algo={'name': 'UnitStep'},
                                       need_init_orbs=False))
    # Ground-state occupation numbers
    f_sn = excite(calc, 0, 0, spin=(0, 0))
    prepare_mom_calculation(calc, H2O, f_sn)

    def rotate_homo_lumo(calc=calc):
        a = 70 / 180 * np.pi
        iters = calc.get_number_of_iterations()
        if iters == 3:
            c = calc.wfs.kpt_u[0].C_nM.copy()
            calc.wfs.kpt_u[0].C_nM[3] = np.cos(a) * c[3] + np.sin(a) * c[4]
            calc.wfs.kpt_u[0].C_nM[4] = np.cos(a) * c[4] - np.sin(a) * c[3]
            counter = calc.wfs.eigensolver.update_ref_orbs_counter
            calc.wfs.eigensolver.update_ref_orbs_counter = iters + 1
            calc.wfs.eigensolver.update_ref_orbitals(calc.wfs,
                                                     calc.hamiltonian,
                                                     calc.density)
            calc.wfs.eigensolver.update_ref_orbs_counter = counter

    calc.attach(rotate_homo_lumo, 1)
    e = H2O.get_potential_energy()

    assert e == pytest.approx(-5.091912426348663, abs=1.0e-4)
