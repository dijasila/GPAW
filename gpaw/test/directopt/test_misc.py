import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.tools import rotate_orbitals, get_a_vec_u
from ase import Atoms
import numpy as np


@pytest.mark.do
def test_rotate_orbitals(in_tmp_dir):
    # H2 molecule:
    d = 0.74
    H2 = Atoms('H2',
               positions=[(0, 0, 0),
                          (0, 0, d)])
    H2.center(vacuum=4.0)

    calc = GPAW(mode=LCAO(),
                basis='sz(dzp)',
                h=0.25,
                occupations={'name': 'fixed-uniform'},
                eigensolver={'name': 'etdm'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                spinpol=False,
                convergence={'density': 1.0e-4,
                             'eigenstates': 4.0e-8})
    H2.calc = calc
    H2.get_potential_energy()

    C_nM_saved = [[-0.53171298, -0.53171298], [ 1.4628662, -1.4628662]]
    assert (np.abs(calc.wfs.kpt_u[0].C_nM - C_nM_saved) < 1.0e-2).all()

    homo = 0
    lumo = 1
    a = 90.0
    rotate_orbitals(calc.wfs, [0, 1], a, 0)

    C_nM_saved = [[1.4628662, -1.4628662], [0.53171298, 0.53171298]]
    assert (np.abs(calc.wfs.kpt_u[0].C_nM - C_nM_saved) < 1.0e-2).all()


def test_get_a_vec_u(in_tmp_dir):
    # H2 molecule:
    d = 0.74
    H2 = Atoms('H2',
               positions=[(0, 0, 0),
                          (0, 0, d)])
    H2.center(vacuum=4.0)

    calc = GPAW(mode=LCAO(),
                basis='dzp',
                h=0.25,
                occupations={'name': 'fixed-uniform'},
                eigensolver={'name': 'etdm'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                spinpol=False,
                convergence={'density': 1.0e-4,
                             'eigenstates': 4.0e-8})
    H2.calc = calc
    H2.get_potential_energy()

    a_vec_u = get_a_vec_u(
        calc.wfs.eigensolver, calc.wfs, [[0, 1], [0, 3], [0, 7]],
        [0.5 * np.pi, np.pi, -0.1 * np.pi], [0, 0, 0])

    a_vec_u_saved = [1.57079633, 0.0, 3.14159265, 0.0, 0.0, 0.0, -0.31415927,
                     0.0, 0.0]

    assert (np.abs(a_vec_u[0] - a_vec_u_saved) < 1.0e-2).all()
