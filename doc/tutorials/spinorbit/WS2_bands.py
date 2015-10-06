from ase.dft.kpoints import get_bandpath
from ase.parallel import paropen
from gpaw import GPAW

layer = GPAW('WS2_gs.gpw', txt=None).atoms

G = [0, 0, 0]
K = [1/3., 1/3., 0]
M = [0.5, 0, 0]
M_ = [-0.5, 0, 0]
K_ = [-1/3., -1/3., 0]
kpts, x, X = get_bandpath([M, K, G, K_, M_], layer.cell, npoints=1000)

calc = GPAW('WS2_gs.gpw', kpts=kpts, symmetry='off')
calc.diagonalize_full_hamiltonian(nbands=100)

calc.write('WS2_bands.gpw', mode='all')

f = paropen('WS2_kpath.dat', 'w')
for k in x:
    print >> f, k
f.close()

f = paropen('WS2_highsym.dat', 'w')
for k in X:
    print >> f, k
f.close()
