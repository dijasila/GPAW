import numpy as np
from ase.units import Bohr, Ha
from ase.build import molecule
from gpaw.cluster import Cluster
from gpaw import GPAW
from gpaw.external import static_polarizability


atoms = Cluster(molecule('H2O'))
atoms.minimal_box(3)
atoms.calc = GPAW(txt=None)

alpha_cc = static_polarizability(atoms)
print('Polarizability tensor (units Angstrom^3):')
print(alpha_cc * Bohr * Ha)

w, v = np.linalg.eig(alpha_cc)
print('Eigenvalues', w * Bohr * Ha)
print('Eigenvectors', v)
print('average polarizablity', w.sum() * Bohr * Ha / 3, 'Angstrom^3')
