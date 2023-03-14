from ase import Atoms
from gpaw import GPAW, PW
h = Atoms('H', cell=(5, 5, 5))
h.center()
for ecut in range(200, 1001, 100):
    h.calc = GPAW(setups='ae',
                  mode=PW(ecut),
                  txt=f'aeH_ecut{ecut}.txt')
    e = h.get_potential_energy()
