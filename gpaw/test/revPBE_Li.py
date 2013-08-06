from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW, PoissonSolver
from gpaw.test import equal

a = 5.0
n = 24
li = Atoms('Li', magmoms=[1.0], cell=(a, a, a), pbc=True)

calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE',
            poissonsolver=PoissonSolver(nn='M'),
            convergence=dict(eigenstates=4.5e-8))
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
equal(e, -7.462 * Hartree, 1.4)

calc.set(xc='revPBE')
erev = li.get_potential_energy() + calc.get_reference_energy()

equal(erev, -7.487 * Hartree, 1.3)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)
