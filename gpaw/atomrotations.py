import numpy as np


class AtomRotations:
    def __init__(self, R_sii):
        self.R_sii = R_sii
        self.ni = R_sii.shape[1]

    @classmethod
    def from_R_slmm(cls, ni, l_j, R_slmm):
        nsym = len(R_slmm)
        R_sii = np.zeros((nsym, ni, ni))
        i1 = 0
        for l in l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lmm in enumerate(R_slmm):
                R_sii[s, i1:i2, i1:i2] = R_lmm[l]
            i1 = i2
        return cls(R_sii)

    def symmetrize(self, a, D_aii, map_sa):
        D_ii = np.zeros((self.ni, self.ni))
        for s, R_ii in enumerate(self.R_sii):
            D_ii += np.dot(R_ii, np.dot(D_aii[map_sa[s][a]],
                                        np.transpose(R_ii)))
        return D_ii / len(map_sa)
