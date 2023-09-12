import numpy as np


class SymmetrizationPlan:
    def __init__(self, rotation_lsmm, a_sa, l_aj, layout):
        ns = a_sa.shape[0]
        na = a_sa.shape[1]
        nl = len(rotation_lsmm)
        cosets = {frozenset(a_sa[:, a]) for a in range(na)}
        nP = [(2 * l + 1)**2 for l in range(nl)]
        S_lsPP = {l: np.einsum('sab,scd->sacbd',
                               rotation_lsmm[l],
                               rotation_lsmm[l]).reshape((ns, nP[l], nP[l]))
                  / ns for l in range(nl)}
        # P = (i, j)  in contrast to p = svec((i,j))
        S_alZZ = {}
        for coset in map(list, cosets):
            nA = len(coset)
            a = coset[0]  # Representative atom for coset
            S_lZZ = {}  # Z = (a, P)
            for l in range(4):
                S_ZZ = np.zeros((nA * nP[l],) * 2)
                for loca1, a1 in enumerate(coset):
                    Z1 = loca1 * nP[l]
                    Z2 = Z1 + nP[l]
                    for s, a2 in enumerate(a_sa[:, a1]):
                        loca2 = coset.index(a2)
                        Z3 = loca2 * nP[l]
                        Z4 = Z3 + nP[l]
                        S_PP = S_lsPP[l][s]
                        S_ZZ[Z1:Z2, Z3:Z4] += S_PP
                S_lZZ[l] = S_ZZ
            S_alZZ[a] = S_lZZ

            l_j = l_aj[coset[0]]
            Itot = sum([2 * l2 + 1 for l2 in l_j])

            work = []
            for j, l in enumerate(l_j):
                indices = []
                for loca1, a1 in enumerate(coset):
                    a1_, start, end = layout.myindices[a1]
                    assert a1_ == a1
                    Istart = sum([2 * l2 + 1 for l2 in l_j[:j]])
                    Iend = Istart + 2 * l + 1
                    for X in range(Istart, Iend):
                        for Y in range(Istart, Iend):
                            indices.append(start + X * Itot + Y)
                work.append((a, l, np.array(indices)))

        self.work = work
        self.S_alZZ = S_alZZ

    def apply(self, source, target):
        for a, l, ind in self.work:
            for spin in range(len(source)):
                target[spin, ind] = self.S_alZZ[a][l] @ source[spin, ind]
