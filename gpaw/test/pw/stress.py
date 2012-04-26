import numpy as np
from ase.structure import bulk
from ase.units import Bohr
from gpaw import GPAW
from gpaw.wavefunctions.pw import PW
from gpaw.stress import stress
from gpaw.xc.kernel import XCNull
from gpaw.xc import XC

si = bulk('Si', 'diamond', cubic=True)
k=2
#del si[:]
si.set_calculator(GPAW(setups='ah',
                       mode=PW(250),
                       #charge=-32,
                       kpts=(k, k, k),
                       #xc=XC(XCNull()),
                       dtype=complex,
                       usesymm=False,
                       txt='si.txt'))
si.get_potential_energy()
e0 = si.calc.hamiltonian.Etot
vol = si.get_volume() / Bohr**3
sigma_cv = si.calc.wfs.get_kinetic_stress() / vol
sigma_cv += si.calc.hamiltonian.stress * np.eye(3) / vol
p0 = stress(si.calc)
eps = 1e-5
si.set_cell(si.cell * (1 + eps), scale_atoms=True)
si.get_potential_energy()
e1 = si.calc.hamiltonian.Etot
print sigma_cv
p = (e1 - e0) / eps / vol
print p0, p
print abs(p - p0)
