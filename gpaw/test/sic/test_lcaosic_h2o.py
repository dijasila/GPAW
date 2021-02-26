from gpaw import GPAW, LCAO, FermiDirac
from ase import Atoms
import numpy as np
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
from gpaw.test import equal


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
                occupations=FermiDirac(width=0.0, fixmagmom=True),
                eigensolver=DirectMinLCAO(
                    odd_parameters={'name': 'PZ_SIC',
                                    'scaling_factor': (0.5, 0.5)  # SIC/2
                                    }
                ),
                mixer={'name': 'dummy'},
                nbands='nao',
                symmetry='off'
                )
    H2O.calc = calc
    e = H2O.get_potential_energy()
    f = H2O.get_forces()

    equal(e, -11.856260, 1e-5)

    f2 = np.array([[-3.27136, -5.34168, -0.00001],
                   [5.13882, -0.17066, 0.00001],
                   [-1.40629, 5.05699, -0.00001]])
    equal(f2, f, 3e-2)

    calc.write('h2o.gpw', mode='all')
    from gpaw import restart
    H2O, calc = restart('h2o.gpw', txt='-')
    H2O.positions += 1.0e-6
    f3 = H2O.get_forces()
    niter = calc.get_number_of_iterations()
    equal(niter, 4, 3)
    equal(f2, f3, 3e-2)
