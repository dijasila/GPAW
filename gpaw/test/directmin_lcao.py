from gpaw import GPAW, LCAO, FermiDirac
from ase import Atoms
import numpy as np
from gpaw.test import equal

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
            nbands='nao',
            symmetry='off'
            )
H2O.set_calculator(calc)
e = H2O.get_potential_energy()
f = H2O.get_forces()

equal(e, -13.643156256566218, 1.0e-5)


f2 = np.array([[-1.11463, -1.23723, 0.0],
               [1.35791, 0.00827, 0.0],
               [-0.34423, 1.33207, 0.0]])
equal(f2, f, 1e-2)

calc.write('h2o.gpw', mode='all')
from gpaw import restart
H2O, calc = restart('h2o.gpw')
H2O.positions += 1.0e-6
f3 = H2O.get_forces()
niter = calc.get_number_of_iterations()
equal(niter, 3, 1)
equal(f2, f, 1e-2)
