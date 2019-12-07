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
    assert np.allclose(alpha_ig.sum(axis=0), 1)

    f_k = fftn(n_g[np.newaxis, :] * alpha_ig)
    res_ig = ifftn(npp_conv(Grs_ik, f_k))
    assert res_ig.shape == alpha_ig.shape
    return res_ig.sum(axis=0)


def V2(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
    assert alpha_ig.shape == Z_ig.shape
    from ffts import fftn, ifftn
    from convolutions import npp_conv

    cross_index_g = (alpha_ig != 0).argmax(axis=0)
    
    ni, nx, ny, nz = Z_ig.shape
    xs, ys, zs = np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
    assert np.allclose(alpha_ig[cross_index_g - 1, xs, ys, zs], 0)
    assert np.allclose(alpha_ig[cross_index_g + 2, xs, ys, zs], 0)
    assert not np.allclose(alpha_ig[cross_index_g, xs, ys, zs], 0)
    
    dZ_g = Z_ig[cross_index_g + 1, xs, ys, zs] - Z_ig[cross_index_g, xs, ys, zs]
    dG_k = Gs_ik[cross_index_g + 1, xs, ys, zs] - Gs_ik[cross_index_g, xs, ys, zs]

    n_k = fftn(n_g)
    Gnconv_g = ifftn(npp_conv(Grs_ik[cross_index_g, xs, ys, zs], n_k))
    assert Gnconv_g.shape == n_g.shape

    res = ifftn(npp_conv(Gs_ik[cross_index_g + 1, xs, ys, zs], fftn(Gnconv_g * n_g / dZ_g)))
    res2 = ifftn(npp_conv(dG_k, fftn(Gnconv_g * n_g * alpha_ig[cross_index_g, xs, ys, zs] / dZ_g)))

    result1 = (res - res2).copy()

    Gnconv_g = ifftn(npp_conv(Grs_ik[cross_index_g + 1, xs, ys, zs], n_k))
    res = ifftn(npp_conv(-Gs_ik[cross_index_g, xs, ys, zs], fftn(Gnconv_g * n_g / dZ_g)))
    res2 = ifftn(npp_conv(dG_k, fftn(Gnconv_g * n_g * alpha_ig[cross_index_g + 1, xs, ys, zs] / dZ_g)))

    result2 = (res - res2).copy()

    return result1 + result2


def V(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik):
    return V1(n_g, alpha_ig, Grs_ik) + V1p(n_g, alpha_ig, Grs_ik) + V2(n_g, alpha_ig, Z_ig, Grs_ik, Gs_ik)
