"""Compare TPSS from scratch and from PBE"""
from ase import Atoms
from gpaw import GPAW, Davidson

n = Atoms('N', magmoms=[3])
n.center(vacuum=2.0)
n.calc = GPAW(xc='TPSS', h=0.25, eigensolver=Davidson(5),
              convergence=dict(energy=5e-6))
e1 = n.get_potential_energy()

n.calc = GPAW(xc='PBE', h=0.25, eigensolver=Davidson(5),
              convergence=dict(energy=5e-6))
n.get_potential_energy()
n.calc.set(xc='TPSS')
e2 = n.get_potential_energy()
print('Energy difference', e1 - e2)
err = abs(e1 - e2)
assert err < 2e-5, err
