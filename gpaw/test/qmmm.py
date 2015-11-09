from math import cos, sin

import numpy as np

from ase import Atoms
from ase.calculators.tip3p import (TIP3P, epsilon0, sigma0, rOH, thetaHOH,
                                   set_tip3p_charges)
from ase.calculators.qmmm import EIQMMM, LJInteractions
from gpaw import GPAW

r = rOH
a = thetaHOH
D = np.linspace(2.5, 3.5, 20)
kwargs = {'h': 0.17}

monomer = Atoms('H2O',
                [(r * cos(a), 0, r * sin(a)),
                 (r, 0, 0),
                 (0, 0, 0)])
monomer.center(vacuum=3.0)
monomer.calc = GPAW(txt='1.txt', **kwargs)
e1 = monomer.get_potential_energy()

dimer = Atoms('H2OH2O',
              [(r * cos(a), 0, r * sin(a)),
               (r, 0, 0),
               (0, 0, 0),
               (r * cos(a / 2), r * sin(a / 2), 0),
               (r * cos(a / 2), -r * sin(a / 2), 0),
               (0, 0, 0)])
set_tip3p_charges(dimer)

interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})
dimer.calc = EIQMMM([0, 1, 2],
                    GPAW(txt='2.txt', **kwargs), TIP3P(),
                    interaction, vacuum=3)
E = []
for d in D:
    dimer.positions[3:, 0] += d - dimer.get_distance(2, 5)
    E.append(dimer.get_potential_energy() - e1)
if 0:
    import matplotlib.pyplot as plt
    plt.plot(D, E)
    plt.show()
