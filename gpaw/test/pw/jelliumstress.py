import numpy as np
from ase import Atoms
from gpaw.wavefunctions.pw import PW
from ase.units import Bohr, Hartree
from gpaw import GPAW, FermiDirac, Mixer
from gpaw.mpi import world
from gpaw.xc.kernel import XCNull
from gpaw.xc import XC
from gpaw.poisson import NoInteractionPoissonSolver


nelectrons = 16
system = Atoms(cell=[(1.7, 0.1, 0.1),
                     (-0.2, 1.7, 0.0),
                     (0.0, -0.2, 2.)],
               pbc=1)

calc = GPAW(mode=PW(250),
            nbands=24,
            xc=XC(XCNull()),
            charge=-nelectrons,
            kpts=(4, 2, 1),
            usesymm=False,
            #basis='sz(dzp)',
            poissonsolver=NoInteractionPoissonSolver())

system.calc = calc
system.get_potential_energy()
sigma_vv = calc.wfs.get_kinetic_stress()
print sigma_vv
E1 = calc.hamiltonian.Etot

deps = 1e-5

cell = system.cell.copy()
calc.set(txt=None)
for v in range(3):
    system.cell = cell
    system.cell[:, v] *= 1.0 + deps
    system.get_potential_energy()
    E2 = calc.hamiltonian.Etot
    s = (E2 - E1) / deps
    print abs(s - sigma_vv[v, v])
    assert abs(s - sigma_vv[v, v]) < 4e-4
for v1 in range(3):
    v2 = (v1 + 1) % 3
    x = np.eye(3)
    x[v1, v2] += deps
    x[v2, v1] += deps
    system.cell = np.dot(cell, x)
    system.get_potential_energy()
    E2 = calc.hamiltonian.Etot
    s = (E2 - E1) / deps / 2
    print abs(s - sigma_vv[v1, v2])
    assert abs(s - sigma_vv[v1, v2]) < 3e-4
