"""Compare TPSS from scratch and from PBE"""
from ase import Atoms
from gpaw import GPAW

if 1:
    n = Atoms('N', magmoms=[3])
    n.center(vacuum=2.5)
    n.calc = GPAW(xc='TPSS')
    e1 = n.get_potential_energy()
    
    n.calc = GPAW(xc='PBE')
    n.get_potential_energy()
    n.calc.set(xc='TPSS')
    e2 = n.get_potential_energy()
    print('Energy difference', e1 - e2)
    assert abs(e1 - e2) < 1e-5

h = Atoms('H', magmoms=[1])
h.center(vacuum=2.5)
h.calc = GPAW(xc='TPSS')
e1 = h.get_potential_energy()

h.calc = GPAW(xc='PBE')
h.get_potential_energy()
h.calc.set(xc='TPSS')
e2 = h.get_potential_energy()
print('Energy difference', e1 - e2)
assert abs(e1 - e2) < 1e-5
