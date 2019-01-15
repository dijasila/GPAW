import numpy as np
from gpaw.utilities.lapack import diagonalize

def loewdin(C_nM, S_MM):

    """
    Loewdin based orthonormalization
    C_nM = sum_m C_nM[m] [1/sqrt(S)]_mn

    S_mn = (C_nM[m].conj(), S_MM C_nM[n])
    """
    S_overlapp = np.dot(C_nM.conj(), np.dot(S_MM, C_nM.T))

    ev = np.zeros(S_overlapp.shape[0], dtype=float)
    diagonalize(S_overlapp, ev)
    ev_sqrt = np.diag(1.0 / np.sqrt(ev))

    S = np.dot(np.dot(S_overlapp.T.conj(), ev_sqrt), S_overlapp)

    return np.dot(S.T, C_nM)