import numpy as np
from gpaw.grid_descriptor import GridDescriptor

N = 40
N_c = (N, N, N)
gd = GridDescriptor(N_c)

nb_i = np.arange(0.6, 2, 0.2) # np.array([0.5])

from splines import build_splines
Gs_i, _, _, _ = build_splines(nb_i, gd)


# Gs = Gs_i[0]

ks = np.arange(0, 100, 0.5)

for i, Gs in enumerate(Gs_i):
    G0 = Gs(0)
    print(f"G0: {G0}, nb: {nb_i[i]}, i:{i}, prod: {G0 * nb_i[i]}")
    # assert np.allclose(G0 * nb_i[i], -1), f"G0: {G0}, nb: {nb_i[i]}, i:{i}"

exit()
G_k = Gs(ks)

import matplotlib.pyplot as plt
plt.plot(ks, G_k)
plt.show()
