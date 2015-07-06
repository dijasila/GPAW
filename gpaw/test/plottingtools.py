import numpy as np

from ase.units import Hartree

from gpaw.response.df import DielectricFunction
from gpaw.response.integrators import TetrahedronIntegrator
from gpaw.response.chi0 import ArrayDescriptor

omega_w = np.array([0.5], float) / Hartree
wd = ArrayDescriptor(omega_w)

calc = 'gsresponse.gpw'
df = DielectricFunction(calc)
chi0 = df.chi0

pd = chi0.get_PWDescriptor([0, 0, 0])
bzk_kv, PWSA = chi0.get_kpoints(pd)

# Initialize integrator
integrator = TetrahedronIntegrator()
td = integrator.tesselate(bzk_kv)

# Integrate interband response
m1 = chi0.nocc1
m2 = chi0.nbands
kd = chi0.calc.wfs.kd
n1 = 0
n2 = chi0.nocc2
eig_kwargs = {'kd': kd, 'm1': m1, 'm2': m2, 'n1': 0,
              'n2': n2, 'pd': pd}

get_eigenvalues = chi0.get_eigenvalues

# Relevant quantities
bzk_kv = td.points
nk = len(bzk_kv)

# Store eigenvalues
deps_Mk = None
for K in range(nk):
    k_v = bzk_kv[K]
    deps_M = get_eigenvalues(k_v, 0, **eig_kwargs)
    if deps_Mk is None:
        deps_Mk = np.zeros((len(deps_M), nk), float)
    deps_Mk[:, K] = deps_M

# Store indices for frequencies
indices_SMi = np.zeros((td.nsimplex, deps_M.shape[0], 2), int)

for s, simplex in enumerate(td.simplices):
    teteps_Mk = deps_Mk[:, simplex]
    emin_M, emax_M = teteps_Mk.min(1), teteps_Mk.max(1)
    i0_M, i1_M = wd.get_index_range(emin_M, emax_M)
    indices_SMi[s, :, 0] = i0_M
    indices_SMi[s, :, 1] = i1_M

omega_w = wd.get_data()
nw = len(wd)

patches_wP = [[] for w in range(nw)]

for S, indices_Mi in enumerate(indices_SMi):
    teteps_Mk = deps_Mk[:, td.simplices[S]]
    tetk_kc = td.points[td.simplices[S]]
    for M, indices_i in enumerate(indices_Mi):
        teteps_k = teteps_Mk[M]
        for iw in range(indices_i[0], indices_i[1]):
            o = omega_w[iw]
            patch = []
            for k1 in range(3):
                for k2 in range(k1 + 1, 4):
                    if teteps_k[k1] < o and teteps_k[k2] > o:
                        a = ((o - teteps_k[k1]) /
                             (teteps_k[k2] - teteps_k[k1]))
                        k_v = tetk_kc[k1] * (1 - a) + tetk_kc[k2] * a
                        patch.append(k_v)
                    elif teteps_k[k1] > o and teteps_k[k2] < o:
                        a = ((o - teteps_k[k2]) /
                             (teteps_k[k1] - teteps_k[k2]))
                        k_v = tetk_kc[k2] * (1 - a) + tetk_kc[k1] * a
                        patch.append(k_v)
            patch_kv = np.array(patch)
            cent_v = patch_kv.sum(0) / len(patch_kv)
            v1_v = patch_kv[1] - patch_kv[0]
            v2_v = patch_kv[2] - patch_kv[0]
            cp_kv = patch_kv - cent_v
            key = np.arctan2(np.dot(cp_kv, v1_v),
                             np.dot(cp_kv, v2_v))
            order = np.argsort(key)
            patches_wP[iw].append(patch_kv[order])

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for patches_P in patches_wP:
    for patch_kv in patches_P:
        if not len(patch_kv):
            continue

ax.add_collection3d(Poly3DCollection(patches_wP[0]))
# plt.plot(patch_kv[:, 0], patch_kv[:, 1], patch_kv[:, 2], 'o')

from scipy.spatial import Delaunay
import itertools

# Check kpoint grid
pd = df.chi0.get_PWDescriptor([0, 0, 0])
bzk_kv, _ = df.chi0.get_kpoints(pd)
tri = Delaunay(bzk_kv)

for simplex in tri.simplices:
    points_pv = tri.points[simplex]
    for (i, p1), (j, p2) in itertools.product(enumerate(points_pv),
                                              enumerate(points_pv)):
        if i < j:
            continue
        p_pv = np.array([p1, p2])
#        ax.plot(p_pv[:, 0], p_pv[:, 1], p_pv[:, 2],
#                'k-', linewidth=0.5)

from gpaw.bztools import get_BZ

bzk_kc, ibzk_kc = get_BZ('gsresponse.gpw')
B_cv = pd.gd.icell_cv * 2 * np.pi
bzk_kv = np.dot(bzk_kc, B_cv)
ax.scatter(bzk_kv[:, 0], bzk_kv[:, 1], bzk_kv[:, 2], 'o')

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

plt.show()
