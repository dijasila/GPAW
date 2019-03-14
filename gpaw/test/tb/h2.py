from ase import Atoms
from gpaw import GPAW
# a = Atoms('H')
a = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]])
if 0:
    # a.center(vacuum=5.2)
    a.calc = GPAW(mode='tb')
else:
    a.center(vacuum=2)
    a.calc = GPAW(mode='lcao')
a.get_potential_energy()
