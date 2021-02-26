from gpaw import GPAW, PW, FermiDirac
from ase import Atoms
import numpy as np
from gpaw.directmin.fdpw.directmin import DirectMin
from gpaw.test import equal
from ase.dft.bandgap import bandgap
from ase.units import Ha


def test_pwsic_h2o(in_tmp_dir):
    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * (104.51 + 2.0)
    eps=0.02
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d + eps, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=5.0)

    calc = GPAW(mode=PW(300, force_complex_dtype=True),
                occupations=FermiDirac(width=0.0, fixmagmom=True),
                eigensolver=DirectMin(
                    odd_parameters={'name': 'PZ_SIC',
                                    'scaling_factor': (0.5, 0.5)  # SIC/2
                                    },
                    g_tol=1.0e-4,
                    maxiter=200
                ),
                mixer={'method': 'dummy'},
                symmetry='off',
                spinpol=True
                )
    H2O.calc = calc
    e = H2O.get_potential_energy()
    f = H2O.get_forces()
    efermi = calc.wfs.fermi_levels[0] * Ha
    gap = bandgap(calc, efermi=efermi)[0]

    equal(e, -9.968738, 1e-5)
    #
    f2 = np.array([[0.07058, -0.37841, 0],
                   [-0.33957,  0.19016, 0],
                   [0.00652, 0.52039, 0]])
    equal(f2, f, 3e-2)

    equal(gap, 9.665, 1e-2)
    #
    calc.write('h2o.gpw', mode='all')
    from gpaw import restart
    H2O, calc = restart('h2o.gpw', txt='-')
    H2O.positions += 1.0e-6
    f3 = H2O.get_forces()
    niter = calc.get_number_of_iterations()
    equal(niter, 4, 3)
    equal(f2, f3, 3e-2)
