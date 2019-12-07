# Function to calculate Z_ig
# i.e. convolution of density with G_i


def calc_Z_ig(n_g, gd, nb_i):
    import numpy as np
    from splines import build_splines
    from convolutions import npp_conv
    from ffts import fftn, ifftn

    K_k = get_K_K(gd)
    # assert np.allclose(np.min(K_k), 0)
    # kmax = np.max(K_k)
    Gs_i, _, _, _ = build_splines(nb_i, gd)

    Gs_iK = np.array([Gs(K_k) for Gs in Gs_i])
    
    n_K = fftn(n_g)
    Z_ig = ifftn(npp_conv(Gs_iK, n_K))

    return Z_ig    


def get_K_K(gd):
    from gpaw.utilities.tools import construct_reciprocal
    K2_K, _ = construct_reciprocal(gd)
    K2_K[0, 0, 0] = 0
    return K2_K**(1 / 2)
