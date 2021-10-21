import numpy as np
from ase.parallel import paropen
from gpaw import GPAW

atoms = GPAW('WS2_gs.gpw', txt=None).atoms

G, K, M = np.array([[0, 0, 0],
                    [1 / 3, 1 / 3, 0],
                    [0.5, 0, 0]])
bp = atoms.cell.bandpath([M, K, G, -K, -M], npoints=1000)

calc = GPAW('WS2_gs.gpw').fixed_density(
    kpts=bp.kpts, symmetry='off')
calc.write('WS2_bands.gpw')

x, X, labels = bp.get_linear_kpoint_axis()

with paropen('WS2_kpath.dat', 'w') as f:
    for k in x:
        print(k, file=f)


with paropen('WS2_highsym.dat', 'w') as f:
    for k in X:
        print(k, file=f)
