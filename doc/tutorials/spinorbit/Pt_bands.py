from ase.dft.kpoints import ibz_points, get_bandpath
from gpaw import GPAW

calc = GPAW('Pt_gs.gpw', txt=None)

points = ibz_points['fcc']
G = points['Gamma']
X = points['X']
W = points['W']
L = points['L']
K = points['K']
kpts, x, X = get_bandpath([G, X, W, L, G, K, X], calc.atoms.cell, npoints=200)

calc = GPAW('Pt_gs.gpw',
            kpts=kpts,
            symmetry='off',
            txt='Pt_bands.txt')
calc.diagonalize_full_hamiltonian(nbands=20)

calc.write('Pt_bands.gpw', mode='all')
