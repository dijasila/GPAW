import numpy as np
from gpaw.grid_descriptor import GridDescriptor

gd = GridDescriptor([4, 4, 4])
a = gd.empty(dtype=complex)
a[:] = 1.0
assert gd.integrate(a.real, a.real) == 1.0

gd = GridDescriptor([10] * 3, cell_cv=[1.] * 3, pbc_c=[1, 0, 0])
# Point outside box in non-periodic direction should stay
r_v = np.array([0, 2.5, 0])
dr_cG = gd.get_grid_point_distance_vectors(r_v)
assert (dr_cG[:, 0, 0, 0] == np.dot(gd.h_cv, gd.beg_c) - r_v).all()
# Point outside box in periodic direction should be folded inside
r_v = np.array([2.5, 0, 0])
dr_cG = gd.get_grid_point_distance_vectors(r_v)
assert (dr_cG[:, 0, 0, 0] == np.dot(gd.h_cv, gd.beg_c) - r_v % 1.0).all()
