import numpy as np



def wda_energy(n_g, alpha_ig, Grs_ik):
    # Convolve n with G/r
    # Multiply result by alpha * n
    # Integrate
    from convolutions import npp_conv
    from ffts import fftn, ifftn
    
    n_k = fftn(n_g)
    res = ifftn(npp_conv(Grs_ik, n_k))
    assert res.shape == alpha_ig.shape

    result_g = np.sum(res * n_g[np.newaxis, :] * alpha_ig, axis=0)

    return result_g
    
    
