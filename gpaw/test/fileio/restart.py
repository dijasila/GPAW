from __future__ import division
from math import sqrt
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, restart
from gpaw.test import equal

d = 3.0
atoms = Atoms('Na3',
              positions=[(0, 0, 0),
                         (0, 0, d),
                         (0, d * sqrt(3 / 4), d / 2)],
              magmoms=[1.0, 1.0, 1.0],
              cell=(3.5, 3.5, 4 + 2 / 3),
              pbc=True)


def test(atoms):
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    m0 = atoms.get_magnetic_moments()
    eig00 = atoms.calc.get_eigenvalues(spin=0)
    eig01 = atoms.calc.get_eigenvalues(spin=1)
    # Write the restart file(s)
    atoms.calc.write('tmp')
    atoms.calc.write('tmp2', 'all')

    # Try restarting from all the files
    atoms, calc = restart('tmp2')

    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    m1 = atoms.get_magnetic_moments()
    eig10 = calc.get_eigenvalues(spin=0)
    eig11 = calc.get_eigenvalues(spin=1)
    print(e0, e1)
    equal(e0, e1, 1e-10)
    print(f0, f1)
    for ff0, ff1 in zip(f0, f1):
        err = np.linalg.norm(ff0 - ff1)
        assert err <= 1e-10
    print(m0, m1)
    for mm0, mm1 in zip(m0, m1):
        equal(mm0, mm1, 1e-10)
    print('A', eig00, eig10)
    for eig0, eig1 in zip(eig00, eig10):
        equal(eig0, eig1, 1e-10)
    print('B', eig01, eig11)
    for eig0, eig1 in zip(eig01, eig11):
        equal(eig0, eig1, 1e-10)

    # Check that after restart everything is writable
    calc.write('tmp3')
    calc.write('tmp4', 'all')


# Only a short, non-converged calcuation
conv = {'eigenstates': 1.24, 'energy': 2e-1, 'density': 1e-1}

atoms.calc = GPAW(mode=PW(200),  # h=0.30,
                  nbands=3,
                  setups={'Na': '1'},
                  convergence=conv)
test(atoms)
atoms.calc = GPAW(h=0.30,
                  nbands=3,
                  setups={'Na': '1'},
                  convergence=conv)
test(atoms)
