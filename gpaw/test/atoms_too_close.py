from ase import Atoms
from gpaw import GPAW

atoms = Atoms('H2', [(0.0, 0.0, 0.0), 
                     (0.0, 0.0, 3.7)], 
              cell=(4, 4, 4), pbc=True)

calc = GPAW(txt=None, atoms=atoms)

try:
    atoms.get_potential_energy()
except RuntimeError:
    pass
else:
    assert 2 + 2 == 5
