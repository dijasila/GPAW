from __future__ import print_function

import numpy as np

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt

from ase.visualize import view
from ase.lattice import bulk

from gpaw.bztools import get_BZ
from gpaw import GPAW


atom = bulk('Na')
atom.calc = GPAW(mode='pw')
atom.center()
atom.get_potential_energy()

atoms = [atom]

for atom in atoms:
    cell_cv = atom.cell
    calc = atom.calc

    bzk_kc, ibzk_kc = get_BZ(calc)

    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T
    bzk_kv = np.dot(bzk_kc, B_cv)
    ibzk_kv = np.dot(ibzk_kc, B_cv)
    bzhull = ConvexHull(bzk_kv)
    hull = ConvexHull(ibzk_kv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(bzk_kv[:, 0], bzk_kv[:, 1], bzk_kv[:, 2])
#    ax.scatter(ibzk_kv[:, 0], ibzk_kv[:, 1], ibzk_kv[:, 2])

    U_scc = atom.calc.wfs.kd.symmetry.op_scc

    k_skv = []
    from gpaw.symmetry import Symmetry
    latsym = Symmetry([0], cell_cv)
    latsym.find_lattice_symmetry()
    from gpaw.bztools import get_symmetry_operations
    symmetry = atom.calc.wfs.kd.symmetry
    cU_scc = get_symmetry_operations(symmetry.op_scc,
                                     symmetry.time_reversal)

    for s, U_cc in enumerate(cU_scc):
        MT_vv = np.dot(B_cv.T, np.dot(U_cc, A_cv))

        ks_skv = []
        for simplex in hull.simplices[::-1]:
            points_pv = hull.points[simplex]
            points_pv = np.dot(points_pv, MT_vv.T)
            ks_skv.append(points_pv)

        tmp_kv = np.concatenate(ks_skv)
        
        kavg_v = tmp_kv.sum(0) / len(tmp_kv)
        if s == 0:
            factor = 1.0
        else:
            factor = 1.0
        k_skv.extend([ks_kv + kavg_v * factor for ks_kv in ks_skv])

    collection = Poly3DCollection(k_skv, linewidths=.5,
                                  antialiaseds=True)

    face_color = [0, 1, 0]
    collection.set_facecolor([1, 1, 1])
    ax.add_collection3d(collection)

    bzk_kv = np.concatenate(k_skv)

    X = bzk_kv[:, 0]
    Y = bzk_kv[:, 1]
    Z = bzk_kv[:, 2]
    max_range = np.array([X.max() - X.min(),
                          Y.max() - Y.min(),
                          Z.max() - Z.min()]).max() / 2.0

    mean_x = X.mean()
    mean_y = Y.mean()
    mean_z = Z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    ax.view_init(elev=36.0, azim=11.0)
    ax.set_aspect('equal')
plt.savefig('/home/morten/bz2.png', bbox_inches='tight')
plt.show()
