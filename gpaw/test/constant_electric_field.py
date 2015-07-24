"""A proton in an electric field."""
from ase import Atoms
from gpaw import GPAW
from gpaw.external import ConstantElectricField

h = Atoms('H')
h.center(vacuum=2.5)
h.calc = GPAW(external=ConstantElectricField(1.0),  # 1 eV / Ang
              charge=1,
              txt='h.txt')
e = h.get_potential_energy()
f1 = h.get_forces()[0, 2]
h[0].z += 0.001
de = h.get_potential_energy() - e
f2 = -de / 0.001
print(f1, f2)
assert abs(f1 - 1) < 1e-4
assert abs(f2 - 1) < 5e-3
