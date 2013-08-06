import os
from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW
from gpaw.test import equal
import gpaw.mpi as mpi
from gpaw.atom.generator2 import generate

# Generate non-scalar-relativistic setup for Cu:
generate(['Cu', '-w'])

a = 8.0
c = a / 2
Cu = Atoms('Cu', [(c, c, c)], magmoms=[1],
           cell=(a, a, a), pbc=0)

calc = GPAW(h=0.2, lmax=0, setups='./')
Cu.set_calculator(calc)
e = Cu.get_potential_energy()
niter = calc.get_number_of_iterations()

e_4s_major = calc.get_eigenvalues(spin=0)[5] / Hartree
e_3d_minor = calc.get_eigenvalues(spin=1)[4] / Hartree
print mpi.rank, e_4s_major, e_3d_minor

#
# The reference values are from:
#
#   http://physics.nist.gov/PhysRefData/DFTdata/Tables/29Cu.html
#
equal(e_4s_major - e_3d_minor, -0.184013 - -0.197109, 0.001)
