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


Gs_i, Grs_i, dGs_i, dGrs_i = build_splines(nb_i, gd)


from ks import get_K_K
K_K = get_K_K(gd)
print(f"K max: {np.max(K_K)}")

plt.subplots(figsize=(20, 20))
for ii, Gs in enumerate(Gs_i):
    ks = np.linspace(0, np.max(K_K)/10, 100)
    plt.subplot(len(nb_i), 1, ii + 1)
    plt.plot(ks, Gs(ks), label=str(nb_i[ii]))
    plt.plot(ks, Grs_i[ii](ks), label=str(nb_i[ii]) + "--1/r")
    plt.legend()
    plt.tight_layout()
plt.savefig("test.png")
