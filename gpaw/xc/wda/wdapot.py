import numpy as np


def V1(n_g, alpha_ig, Grs_ik):
    from ffts import fftn, ifftn
    from convolutions import npp_conv

    n_k = fftn(n_g)
    
    res_ig = ifftn(npp_conv(Grs_ik, n_k))
    assert res_ig.shape == alpha_ig.shape

    result_g = np.sum(res_ig * alpha_ig, axis=0)
    return result_g


def V1p(n_g, alpha_ig, Grs_ik):
    from ffts import fftn, ifftn
    from convolutions import npp_conv

    f_k = fftn(n_g[np.newaxis, :] * alpha_ig)
    res_ig = ifftn(npp_conv(Grs_ik, f_k))
    assert res_ig.shape == alpha_ig.shape

    return res_ig.sum(axis=0)


def V2(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
    assert alpha_ig.shape == Z_ig.shape
    from ffts import fftn, ifftn
    from convolutions import npp_conv

    dZ_ig = np.roll(Z_ig, -1, axis=0) - Z_ig
    n_k = fftn(n_g)
    Gnconv_ig = fftn(npp_conv(Grs_ik, n_k))
    res = ifftn(npp_conv(np.roll(Gs_ik, -1, axis=0), fftn(Gnconv_ig * n_g[np.newaxis, :] / dZ_ig)))

    res2 = ifftn(npp_conv(np.roll(Gs_ik, -1, axis=0) - Gs_ik, fftn(Gnconv_ig * n_g[np.newaxis, :] * alpha_ig / dZ_ig)))
    
    result = ((res - res2) * (alpha_ig != 0)).sum(axis=0)
    
    return result


def V(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
    return (V1(n_g, alpha_ig, Grs_ik) 
            + V1p(n_g, alpha_ig, Grs_ik)
            + V2(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik)
            )
