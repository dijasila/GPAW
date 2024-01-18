import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW
import numpy as np


def test_direct_optimization():
    """test with 4 cores to check parallelization over
     both kpts and domain"""

    a = 2.0
    atoms = Atoms('Li',
                  cell=[a, a, a],
                  pbc=True)
    atoms.calc = GPAW(
        mode='pw',
        kpts=(1, 1, 2),
        nbands=-2,
        symmetry="off",
        mixer={'backend': 'no-mixing'},
        occupations={"name": "fixed-uniform"},
        eigensolver={"name": "lbfgs", "memory": 2},
        convergence={"bands": "occupied"},
    )
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(-2.3329, rel=0.05)


@pytest.mark.do
def test_directmin_pw(in_tmp_dir):

    atoms = Atoms('CCHHHH',
                  positions=[
                      [-0.66874198, -0.00001714, -0.00001504],
                      [0.66874210, 0.00001699, 0.00001504],
                      [-1.24409879, 0.00000108, -0.93244784],
                      [-1.24406253, 0.00000112, 0.93242153],
                      [1.24406282, -0.93242148, 0.00000108],
                      [1.24409838, 0.93244792, 0.00000112]
                  ]
                  )
    atoms.center(vacuum=4.0)
    atoms.set_pbc(False)

    calc = GPAW(mode={'name': 'pw', 'force_complex_dtype': True, 'ecut': 300},
                xc='PBE',
                occupations={'name': 'fixed-uniform'},
                eigensolver={"name": "lbfgs", "memory": 3},
                mixer={'backend': 'no-mixing'},
                spinpol=True,
                symmetry='off',
                nbands=-5,
                convergence={'eigenstates': 4.0e-6, 'bands': 7},
                )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    f = atoms.get_forces()
    fsaved = np.array([[-3.73061, 0.00020, -0.00011],
                       [3.72978, 0.00002, -0.00020],
                       [-0.61951, -2.63437, -0.47774],
                       [-0.62006, 2.63439, 0.47806],
                       [0.62080, -0.47803, -2.63261],
                       [0.62030, 0.47779, 2.63259]])

    assert fsaved == pytest.approx(f, abs=1e-2)
    assert energy == pytest.approx(-26.205455, abs=1.0e-4)
    assert calc.wfs.kpt_u[0].eps_n[5] > calc.wfs.kpt_u[0].eps_n[6]

    # calc.write('ethylene.gpw', mode='all')
    # calc = GPAW('ethylene.gpw', txt='-')
    # atoms = calc.get_atoms()
    # atoms.positions += 1.0e-6
    # f2 = atoms.get_forces()
    # niter = calc.get_number_of_iterations()
    #
    # assert niter == pytest.approx(3, abs=1)
    # assert fsaved == pytest.approx(f2, abs=1e-2)
