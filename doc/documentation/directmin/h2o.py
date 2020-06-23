from gpaw import GPAW, LCAO, FermiDirac
from ase import Atoms
import numpy as np

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
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            eigensolver='direct_min_lcao',
            mixer={'method': 'dummy'},
            nbands='nao'
            )
H2O.set_calculator(calc)
H2O.get_potential_energy()
