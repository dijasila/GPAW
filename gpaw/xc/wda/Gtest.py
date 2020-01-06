import numpy as np
from splines import build_splines
import matplotlib.pyplot as plt
from gpaw.grid_descriptor import GridDescriptor
N = 50
gd = GridDescriptor(N_c=(N, N, N))

grid_vg = gd.get_grid_point_coordinates()
v = grid_vg[:, N//2, N//2, N//2]
r_g = np.linalg.norm(grid_vg - v[:, np.newaxis, np.newaxis, np.newaxis], axis=0)

md = np.max(r_g)
#n_g = np.random.rand(*N_c) * 100
n_g = np.exp(-r_g**2 * 10 / md**2 )

from nbars import get_nbars
nnbar = 20
nb_i = get_nbars(n_g, npts=nnbar)


from splines import C, lambd
from ks import get_K_K
K_K = get_K_K(gd)
kmax = 1.2 * np.max(K_K)

dr = 0.001
rmax = 100
r_j = np.arange(dr / 5, rmax, dr)

nks = 900
k_k = np.exp(np.linspace(0, np.log(kmax), nks)) - 1
assert np.allclose(np.min(k_k), 0)
na = np.newaxis
r_ikj = r_j[na, na, :]
k_ikj = k_k[na, :, na]

C_i, dC_i = C(nb_i)
C_ikj = C_i[:, na, na]
dC_ikj = dC_i[:, na, na]

lambd_i, dlambd_i = lambd(nb_i)
lambd_ikj = lambd_i[:, na, na]
dlambd_ikj = dlambd_i[:, na, na]

G_ikj = C_ikj * (1 - np.exp(-(lambd_ikj / r_ikj)**5))

plt.subplots(figsize=(20,20))
for ii, G_kj in enumerate(G_ikj):
    assert G_kj.shape[0] == 1
    plt.subplot(len(nb_i), 1, ii + 1)
    plt.plot(r_j, G_kj[0, :], label=str(nb_i[ii]))
    plt.legend()
    plt.tight_layout()
plt.savefig("Gtest.png")
