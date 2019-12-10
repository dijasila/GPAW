# Function to calculate Z_ig
# i.e. convolution of density with G_i


def calc_Z_ig(n_g, Gs_iK): # gd, nb_i):
    import numpy as np
    from splines import build_splines
    from convolutions import npp_conv
    from ffts import fftn, ifftn

    # K_k = get_K_K(gd)
    # assert np.allclose(np.min(K_k), 0)
    # kmax = np.max(K_k)
    # Gs_i, _, _, _ = build_splines(nb_i, gd)

    # Gs_iK = np.array([Gs(K_k) for Gs in Gs_i])
    
    n_K = fftn(n_g)
    Z_ig = ifftn(npp_conv(Gs_iK, n_K))
    
    # assert np.allclose(Z_ig[-1], n_g), f"MAE: {np.mean(np.abs(Z_ig[-1], n_g))}"

    return Z_ig    
