import numpy as np

def get_alphas(Z_ig):
    Z_ri = Z_ig.reshape(len(Z_ig), -1).T

    alpha_ri = np.zeros_like(Z_ri)
    for ir, Z_i in enumerate(Z_ri):
        for ii, Z in enumerate(Z_i[:-1]):
            did_cross = (Z_i[ii] <= -1 and Z_i[ii + 1] > -1) \
                        or (Z_i[ii] > -1 and Z_i[ii + 1] <= -1)
            if did_cross:
                alpha_ri[ir, ii] = (Z_i[ii + 1] - (-1)) / (Z_i[ii + 1] - Z_i[ii])
                alpha_ri[ir, ii + 1] = ((-1) - Z_i[ii]) / (Z_i[ii + 1] - Z_i[ii])
                break


    res = alpha_ri.T.reshape(Z_ig.shape)    
    assert np.allclose(res.sum(axis=0), 1)
    return res
