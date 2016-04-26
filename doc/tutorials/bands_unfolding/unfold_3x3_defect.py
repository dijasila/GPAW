import numpy as np

from ase.dft.kpoints import get_bandpath
from gpaw import GPAW
from gpaw.unfold import Unfold, find_K_from_k


a = 3.184
L = 15.
PC = a * np.array([(1., 0., 0),
                 (-1. / 2, np.sqrt(3.) / 2., 0.),
                 (0, 0, L)])
G = np.array([0., 0., 0.])
M = np.array([1 / 2., 0., 0.])
K = np.array([1 / 3., 1 / 3., 0])
path = [M, K, G]
kpts, x, X = get_bandpath(path, PC, 48)

M_SP = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 1.]])

Kpts = []
for k in kpts:
    K = find_K_from_k(k, M_SP)[0]
    Kpts.append(K)


calc = 'gs_3x3_defect.gpw'
calc_bands = GPAW(calc,
                  fixdensity=True,
                  kpts=Kpts,
                  symmetry='off',
                  nbands=220,
                  convergence={'bands': 200})

calc_bands.get_potential_energy()
calc_bands.write('bands_3x3_defect.gpw', 'all')


calc = 'bands_3x3_defect.gpw'
unfold = Unfold(name='3x3_defect',
                calc=calc,
                M=M_SP,
                spinorbit=False)

unfold.spectral_function(kpoints=kpts)
