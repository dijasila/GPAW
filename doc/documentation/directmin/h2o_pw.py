import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.directmin.fdpw.directmin import DirectMin

# Water molecule:
d = 0.9575
t = np.pi / 180 * 104.51
H2O = Atoms(
    "OH2", positions=[(0, 0, 0), (d, 0, 0), (d * np.cos(t), d * np.sin(t), 0)]
)
H2O.center(vacuum=5.0)

calc = GPAW(
    mode="pw",
    occupations={"name": "fixed-uniform"},
    eigensolver=DirectMin(convergelumo=False),
    mixer={"name": "dummy"},
    spinpol=True
)
H2O.set_calculator(calc)
H2O.get_potential_energy()
