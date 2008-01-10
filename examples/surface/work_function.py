from ase import *
import pylab as p
from gpaw import Calculator

calc = Calculator('slab-4.gpw', txt=None)
slab = calc.get_atoms()
L = slab.get_cell()[2, 2]
v = calc.hamiltonian.vt_sG[0] * Hartree
nx, ny, nz = v.shape
z = linspace(0, L, nz)
efermi = calc.get_fermi_level()
p.plot(z, v.sum(axis=0).sum(axis=0) / (nx * ny))
p.plot([0, L], [efermi, efermi])
p.show()

