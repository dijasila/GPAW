class SymmetrizationPlan:
    def __init__(self, xp, rotations, a_sa, l_aj, layout):
        ns = a_sa.shape[0]
        na = a_sa.shape[1]
        cosets = {frozenset(a_sa[:, a]) for a in range(na)}
        S_aZZ = {}
        work = []
        for coset in map(list, cosets):
            nA = len(coset)
            a = coset[0]  # Representative atom for coset
            R_sii = xp.asarray(rotations(l_aj[a], xp))
            i2 = R_sii.shape[1]**2
            R_sPP = xp.einsum('sab,scd->sacbd', R_sii, R_sii)
            R_sPP = R_sPP.reshape((ns, i2, i2)) / ns
            S_ZZ = xp.zeros((nA * i2,) * 2)
            for loca1, a1 in enumerate(coset):
                Z1 = loca1 * i2
                Z2 = Z1 + i2
                for s, a2 in enumerate(a_sa[:, a1]):
                    loca2 = coset.index(a2)
                    Z3 = loca2 * i2
                    Z4 = Z3 + i2
                    S_ZZ[Z1:Z2, Z3:Z4] += R_sPP[s]
            S_aZZ[a] = S_ZZ
            indices = []
            for loca1, a1 in enumerate(coset):
                a1_, start, end = layout.myindices[a1]
                assert a1_ == a1
                for X in range(i2):
                    indices.append(start + X)
            work.append((a, xp.array(indices)))

        self.work = work
        self.S_aZZ = S_aZZ
        self.xp = xp

    def apply(self, source, target):
        total = 0
        for a, ind in self.work:
            for spin in range(len(source)):
                total += len(ind)
                target[spin, ind] = self.S_aZZ[a] @ source[spin, ind]
        assert total == source.shape[1]
