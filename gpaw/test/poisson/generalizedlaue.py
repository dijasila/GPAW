from gpaw.poisson import GeneralizedLauePoissonSolver, FDPoissonSolver
from gpaw.grid_descriptor import GridDescriptor
import numpy as np
gd = GridDescriptor((10,12,42), (4, 5, 20), pbc_c=(True,True,False))
poisson = GeneralizedLauePoissonSolver(nn=2)
poisson.set_grid_descriptor(gd)

poisson2 = FDPoissonSolver(nn=2, eps=1e-28)
poisson2.set_grid_descriptor(gd)

phi_g = gd.zeros()
phi2_g = gd.zeros()
rho_g = gd.zeros()
rho_g[4,5,6] = 1.0
rho_g[4,5,7] = -1.0
poisson.solve(phi_g, rho_g)
poisson2.solve(phi2_g, rho_g)
print "this", phi_g[4,5,:]
print "ref", phi2_g[4,5,:]
print "diff", phi_g[4,5,:]-phi2_g[4,5,:]
assert np.linalg.norm(phi_g-phi2_g) < 1e-10
