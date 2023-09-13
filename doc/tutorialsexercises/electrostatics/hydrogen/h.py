from ase import Atoms
from gpaw import GPAW, PW
h = Atoms('H', cell=(5, 5, 5))
h.center()
for ecut in range(200, 1001, 100):
    h.calc = GPAW(setups='ae',
                  mode=PW(ecut),
                  txt=f'H-{ecut}-ae.txt')
    e = h.get_potential_energy()
