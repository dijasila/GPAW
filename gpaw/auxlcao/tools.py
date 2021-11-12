import numpy as np
from ase.io.cube import write_cube


def density_matrix_to_grid(wfs, rho_MM, q):
    rho_G = wfs.gd.zeros()
    wfs.basis_functions.construct_density(rho_MM, rho_G, q)
    return rho_G

def orbital_product_to_cube(filename, atoms, wfs, M, N):
    nao = wfs.setups.nao
    rho_MM = np.zeros( (nao, nao) )
    rho_MM[M,N] = 1.0
    rho_MM = (rho_MM + rho_MM.T) / 2
    rho_G = density_matrix_to_grid(wfs, rho_MM, 0)
    write_cube(open(filename,'w'), atoms, data=rho_G, comment='')
