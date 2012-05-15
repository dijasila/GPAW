import numpy as np
from ase.lattice import bulk
from ase.units import Bohr, Hartree
from gpaw import GPAW
from gpaw.wavefunctions.pw import PW
from gpaw.stress import stress
from gpaw.xc.kernel import XCNull
from gpaw.xc import XC

si = bulk('Si') * (2, 1, 1)
k = 4
si.calc = GPAW(mode=PW(350),
               kpts=(k // 2, k, k),
               xc='PBE',
               usesymm=False,
               convergence={'energy': 1e-8},
               txt='si.txt')
e0 = si.calc.get_potential_energy(si, force_consistent=True)
p0 = si.get_stress()[:3].sum()
eps = 1e-7
si.set_cell(si.cell * (1 + eps), scale_atoms=True)
e1 = si.calc.get_potential_energy(si, force_consistent=True)
p = (e1 - e0) / eps / si.get_volume()
print p0, p, p - p0
assert abs(p - p0) < 1e-4
