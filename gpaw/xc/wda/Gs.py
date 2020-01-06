import numpy as np

def get_G_ks(Gs_i, Grs_i, k_k):
    Gs_ik = np.array([Gs(k_k) for Gs in Gs_i])
    Grs_ik = np.array([Gs(k_k) for Gs in Gs_i])

    return Gs_ik, Grs_ik
