from ase import Atom, Atoms
from ase.calculators.test import numeric_force
from gpaw import GPAW, Mixer
from gpaw.test import equal

a = 4.0
n = 16
atoms = Atoms([Atom('H', [1.234, 2.345, 3.456])],
                    cell=(a, a, a), pbc=True)
calc = GPAW(nbands=1,
            gpts=(n, n, n),
            txt=None,
            mixer=Mixer(0.25, 3, 1),
            convergence={'energy': 1e-7})
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()
niter1 = calc.get_number_of_iterations()
f1 = atoms.get_forces()[0]
for i in range(3):
    f2i = numeric_force(atoms, 0, i)
    print f1[i]-f2i
    equal(f1[i], f2i, 0.00045)
