# Make a density
# Get Z_ig
# Get alpha_ig
# Get e_g
# Get v_g


import numpy as np
from gpaw.grid_descriptor import GridDescriptor

N = 40
N_c = (N, N, N)
cell_cv = np.array([[2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2]])

gd = GridDescriptor(N_c, cell_cv)


grid_vg = gd.get_grid_point_coordinates()
v = grid_vg[:, N//2, N//2, N//2]
r_g = np.linalg.norm(grid_vg - v[:, np.newaxis, np.newaxis, np.newaxis], axis=0)

md = np.max(r_g)
#n_g = np.random.rand(*N_c) * 100
n_g = np.exp(-r_g**2 * 10 / md**2 )
assert n_g.shape == (N, N, N)

n_g = 4 * n_g / gd.integrate(n_g)
assert (n_g >= 0).all()

from nbars import get_nbars
nnbar = 100
nb_i = get_nbars(n_g, npts=nnbar)


from splines import build_splines
from ks import get_K_K
K_K = get_K_K(gd)

Gs_i, Grs_i, dGs_i, dGrs_i = build_splines(nb_i, gd) # Error is dependent on this step

from Gs import get_G_ks
Gs_ik, Grs_ik = get_G_ks(Gs_i, Grs_i, K_K)


from Zis import calc_Z_ig

Z_ig = calc_Z_ig(n_g, Gs_ik)
print(f"mean Z_ig: {Z_ig.reshape(nnbar, -1).mean(axis=-1)}")
print(f"std Z_ig: {Z_ig.reshape(nnbar, -1).std(axis=-1)}")
assert (Z_ig >= -1).any() and (Z_ig < -1).any()
from alphas import get_alphas

alpha_ig = get_alphas(Z_ig)
assert np.allclose(alpha_ig.sum(axis=0), 1)
print(f"Mean alphas: {alpha_ig.reshape(len(alpha_ig), -1).mean(axis=-1)}")
print(f"std alphas: {alpha_ig.reshape(len(alpha_ig), -1).std(axis=-1)}")
assert (alpha_ig >= 0).all()



from wdaenergy import wda_energy
e_g = wda_energy(n_g, alpha_ig, Grs_ik)


from wdapot import V

v_g = V(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik)

nx, ny, nz = v_g.shape
ix = np.random.randint(nx)
iy = np.random.randint(ny)
iz = np.random.randint(nz)

dn = 0.001

deltan = np.zeros_like(n_g)
deltan[ix, iy, iz] = dn
# e1 = gd.integrate(wda_energy(n_g + deltan, alpha_ig, Grs_ik))
e1 = np.sum(wda_energy(n_g + deltan, alpha_ig, Grs_ik) * gd.dv)
e2 = gd.integrate(wda_energy(n_g - deltan, alpha_ig, Grs_ik))
dedn = (e1 - e2) / (2 * dn * gd.dv)

print("ix, iy, iz:", ix, iy, iz)
print(f"dedn: {dedn}\nv_g: {v_g[ix, iy, iz]}")
# print(f"dv: {gd.dv}, N: {gd.N_c}")
assert np.allclose(dedn, v_g[ix, iy, iz]), f"dedn = {dedn}, v_g = {v_g[ix, iy, iz]}\nv_g/dedn: {v_g[ix, iy, iz] / dedn}\ndv: {gd.dv}\n1/dv: {1/gd.dv}\nnpts: {np.prod(v_g.shape)}"


print("All tests passed")
