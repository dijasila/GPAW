from ase import *
from ase.dft import Wannier
from gpaw import GPAW
from gpaw.test import equal

# Test of ase wannier using gpaw

calc = GPAW(nbands=4)
atoms = molecule('H2', calculator=calc)
atoms.center(vacuum=3.)
e = atoms.get_potential_energy()
niter = calc.get_number_of_iterations()

pos = atoms.positions + np.array([[0, 0, .2339], [0, 0, -.2339]])
com = atoms.get_center_of_mass()

wan = Wannier(nwannier=2, calc=calc, initialwannier='bloch')
equal(wan.get_functional_value(), 2.964, 1e-3)
equal(np.linalg.norm(wan.get_centers() - [com, com]), 0, 1e-4)

wan = Wannier(nwannier=2, calc=calc, initialwannier='projectors')
equal(wan.get_functional_value(), 3.100, 1e-3)
equal(np.linalg.norm(wan.get_centers() - pos), 0, 1e-3)

wan = Wannier(nwannier=2, calc=calc, initialwannier=[[0, 0, .5], [1, 0, .5]])
equal(wan.get_functional_value(), 3.100, 1e-3)
equal(np.linalg.norm(wan.get_centers() - pos), 0, 1e-3)

wan.localize()
equal(wan.get_functional_value(), 3.100, 1e-3)
equal(np.linalg.norm(wan.get_centers() - pos), 0, 1e-3)
equal(np.linalg.norm(wan.get_radii() - 1.2393), 0, 1e-4)
eig = np.sort(np.linalg.eigvals(wan.get_hamiltonian().real))
equal(np.linalg.norm(eig - calc.get_eigenvalues()[:2]), 0, 1e-4)

energy_tolerance = 0.00005
niter_tolerance = 0
equal(e, -6.65697866959, energy_tolerance) # svnversion 5252
#equal(niter, 16, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 16 <= niter <= 17, niter
