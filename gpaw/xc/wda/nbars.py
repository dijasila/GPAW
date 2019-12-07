import numpy as np

def get_nbars(n_g, npts=100):
    min_dens = np.min(n_g)
    max_dens = np.max(n_g)
    
    nb_i = np.exp(np.linspace(np.log(min_dens), np.log(max_dens), npts))
    
    return nb_i
