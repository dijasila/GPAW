from itertools import product, permutations

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ase import Atoms
from ase.units import Hartree

from gpaw import GPAW
from gpaw.bztools import get_BZ, tesselate_brillouin_zone

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import Delaunay, ConvexHull

if True:
    # Calculate GS
    a = 2.456
    cell_cv = a / 2 * np.array([[1, 3**0.5, 0],
                                [1, -3**0.5, 0],
                                [0, 0, 4]])

    atoms = Atoms('CC', positions=[[0, 0, 0],
                                   [0.5, 0.5, 0]],
                  cell=cell_cv, pbc=True)
    atoms.center()
elif True:
    from ase.lattice import bulk
    atoms = bulk('Na')
    atoms.center()
atoms.calc = GPAW(mode='pw', kpts={'size': (4, 4, 4)})
atoms.get_potential_energy()
atoms.calc.write('gs.gpw', 'all')

cell_cv = atoms.cell

# Make refined kpoint grid
rk_kc = tesselate_brillouin_zone('gs.gpw', 1)
B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
rk_kv = np.dot(rk_kc, B_cv)
tri = Delaunay(rk_kv)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

bzk_kc, ibzk_kc = get_BZ('gs.gpw')
bzk_kv = np.dot(bzk_kc, B_cv)
hull = ConvexHull(bzk_kv)

k_skv = []
for simplex in hull.simplices:
    k_skv.append(hull.points[simplex])

pc = Poly3DCollection(k_skv, facecolor=[1, 1, 1],
                      alpha=1.0, linewidths=0.5)

ax.add_collection(pc)

ibzk_kv = np.dot(ibzk_kc, B_cv)

for simplex in tri.simplices:
    points_pv = tri.points[simplex]
    for (i, p1), (j, p2) in product(enumerate(points_pv),
                                    enumerate(points_pv)):
        if i <= j:
            continue
        p_pv = np.array([p1, p2])
        ax.plot(p_pv[:, 0], p_pv[:, 1], p_pv[:, 2], 'k-', linewidth=0.5)

#ax.scatter(rk_kv[:, 0], rk_kv[:, 1], rk_kv[:, 2], 'o')
ax.scatter(bzk_kv[:, 0], bzk_kv[:, 1], bzk_kv[:, 2], '.')
#ax.scatter(ibzk_kv[:, 0], ibzk_kv[:, 1], ibzk_kv[:, 2], 'o')

plt.savefig('/home/morten/tesselate.png',
            bbox_inches='tight')
plt.show()
