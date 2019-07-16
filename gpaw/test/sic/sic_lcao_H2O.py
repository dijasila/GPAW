from gpaw import GPAW, LCAO, FermiDirac
from ase import Atoms
import numpy as np
from gpaw.directmin.directmin_lcao import DirectMinLCAO
from gpaw.test import equal

# Water molecule:
d = 0.9575
t = np.pi / 180 * 104.51
H2O = Atoms('OH2',
            positions=[(0, 0, 0),
                       (d, 0, 0),
                       (d * np.cos(t), d * np.sin(t), 0)])
H2O.center(vacuum=5.0)

calc = GPAW(mode=LCAO(force_complex_dtype=True),
            basis='dzp',
            convergence={'eigenstates': 4.0e-5},
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            eigensolver=DirectMinLCAO(
                odd_parameters={'name': 'PZ_SIC',  # half-SIC
                                'scaling_factor': (0.5, 0.5)}),
            mixer={'method': 'dummy'},
            nbands='nao'
            )
H2O.set_calculator(calc)
e = H2O.get_potential_energy()
f = H2O.get_forces()

equal(e, -16.357, 2e-3)

f2 = np.array([[-0.253, 0.014, 0.009],
               [0.259, -0.095, -0.004],
               [-0.188, 0.220, -0.004]])
equal(f2, f, 3e-2)