from __future__ import print_function

import numpy as np

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt

from gpaw.bztools import get_lattice_BZ


C_cv = [[0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1]]

BCC_cv = [[-1, 1, 1],
          [1, -1, 1],
          [1, 1, -1]]

FCC_cv = [[0, 1, 1],
          [1, 0, 1],
          [1, 1, 0]]

cubic_cells_cv = [C_cv, BCC_cv, FCC_cv]

TET_cv = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 2]]

BCT1_cv = [[-1, 1, 0.5],
           [1, -1, 0.5],
           [1, 1, -0.5]]

BCT2_cv = [[-1, 1, 2],
           [1, -1, 2],
           [1, 1, -2]]

tetragonal_cells_cv = [TET_cv, BCT1_cv, BCT2_cv]

ORC_cv = [[1, 0, 0],
          [0, 2, 0],
          [0, 0, 3]]

ORFC1_cv = [[0, 3, 3],
            [1, 0, 3],
            [1, 3, 0]]

ORFC2_cv = [[0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]

ORFC3_cv = [[0, 2**0.5, 2**0.5],
            [1, 0, 2**0.5],
            [1, 2**0.5, 0]]

ORBC_cv = [[-1, 2, 3],
           [1, -2, 3],
           [1, 2, -3]]

ORCC_cv = [[1, -2, 0],
           [1, 2, 0],
           [0, 0, 3]]

orthorhombic_cells_cv = [ORC_cv, ORFC1_cv, ORFC2_cv,
                         ORFC3_cv, ORBC_cv, ORCC_cv]

HEX_cv = [[1, -3**0.5, 0],
          [1, 3**0.5, 0],
          [0, 0, 1]]

hexagonal_cells_cv = [HEX_cv]

alpha = np.pi / 4

RH1_cv = [[np.cos(alpha / 2.), -np.sin(alpha / 2.), 0],
          [np.cos(alpha / 2.), np.sin(alpha / 2.), 0],
          [np.cos(alpha) / np.cos(alpha / 2.), 0,
           (1 - np.cos(alpha)**2 / np.cos(alpha / 2.)**2)**0.5]]

alpha = np.pi * 3. / 5

d = np.sqrt(np.abs(1 - np.cos(alpha)**2 / np.cos(alpha / 2.)**2))
v = [np.cos(alpha) / np.cos(alpha / 2.), 0, d]

v /= np.linalg.norm(v)

RH2_cv = [[np.cos(alpha / 2.), -np.sin(alpha / 2.), 0],
          [np.cos(alpha / 2.), np.sin(alpha / 2.), 0],
          v.tolist()]

rhombohedral_cells_cv = [RH1_cv, RH2_cv]

monoclinic_cells_cv = []

alpha = np.pi * 1. / 5

MCL_cv = [[0.5, 0, 0],
          [0, 1.0, 0],
          [0, 1.1 * np.cos(alpha), 1.1 * np.sin(alpha)]]
monoclinic_cells_cv.append(MCL_cv)

for a, b, c, alpha in [(0.9, 1.0, 1.1, np.pi / 5)]:
    MCL_cv = [[a / 2, b / 2, 0],
              [-a / 2, b / 2, 0],
              [0, c * np.cos(alpha), c * np.sin(alpha)]]
    monoclinic_cells_cv.append(MCL_cv)

triclinic_cells_cv = []

cubic_names = ['Simple cubic',
               'Body-centered cubic',
               'Face-centered cubic']
tetragonal_names = ['Simple tetragonal',
                    'Body-centered tetragonal: c < a',
                    'Body-centered tetragonal: c > a']
orthorhombic_names_cv = ['Simple orthorhombic',
                         'Face-centered orthorhombic: ' +
                         '1 / a^2 > 1 / b^2 + 1 / c^2',
                         'Face-centered orthorhombic: ' +
                         '1 / a^2 < 1 / b^2 + 1 / c^2',
                         'Face-centered orthorhombic: ' +
                         '1 / a^2 = 1 / b^2 + 1 / c^2',
                         'Body-centered orthorhombic',
                         'C-centered orthorhombic']
hexagonal_names_cv = ['Hexagonal']
rhombohedral_names_cv = ['Rhombohedral: $\\alpha < \\pi / 2$',
                         'Rhombohedral: $\\alpha > \\pi / 2$']

monoclinic_names_cv = ['Simple monoclinic',
                       'C-centered monoclinic']

for cell_cv in cubic_cells_cv:
    print(cell_cv)
    bzk_kc, ibzk_kc, bzedges_lkc, bzfaces_fkc = get_lattice_BZ(cell_cv)

    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T
    E_cv = A_cv / np.linalg.norm(A_cv, axis=1)[:, np.newaxis]

    bzk_kv = np.dot(bzk_kc, B_cv)
    ibzk_kv = np.dot(ibzk_kc, B_cv)
    bzedges_lkv = np.dot(bzedges_lkc, B_cv)

    hull = ConvexHull(ibzk_kv)

    k_kv = ibzk_kv[(ibzk_kv != 0.0).any(1)]
    k_kv /= np.linalg.norm(k_kv, axis=1)[:, np.newaxis]
    e_cv = A_cv / np.linalg.norm(A_cv, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for bzedge_kv in bzedges_lkv:
        ax.plot(bzedge_kv[:, 0], bzedge_kv[:, 1], bzedge_kv[:, 2], '-k')

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

    ax.view_init(elev=30.0, azim=-10.0)
    ax.set_aspect('equal')

    k_skv = []
    for simplex in hull.simplices[::-1]:
        k_skv.append(np.array(ibzk_kv)[simplex])
    
    collection = Poly3DCollection(k_skv, linewidths=1.0)
    face_color = [0, 1, 0]
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)

plt.show()
