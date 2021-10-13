from ase.build import bulk
from gpaw import GPAW, PW

atoms = bulk('Au')

k = 4
calc = GPAW(mode='pw',     
            kpts=(k, k, k))

atoms.calc = calc

energy = atoms.get_potential_energy()
calc.write('au.gpw', mode='all')
