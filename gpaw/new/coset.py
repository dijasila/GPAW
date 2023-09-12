import numpy as np

class SymmetrizationPlan:
    def __init__(self, rotation_lsmm, a_sa, setups, layout):
        ns = a_sa.shape[0]
        na = a_sa.shape[1]
        nl = len(rotation_lsmm)
        cosets = {frozenset(a_sa[:, a]) for a in range(na)}
        nP = [(2 * l + 1)**2 for l in range(nl)]
        S_lsPP = {l: np.einsum('sab,scd->sacbd', rotation_lsmm[l], rotation_lsmm[l]).reshape((ns, nP[l], nP[l])) / ns for l in range(nl)}
        # P = (i, j)  in contrast to p = svec((i,j))
        S_alZZ = {}
        for coset in map(list, cosets):
            nA = len(coset)
            a = coset[0] # Representative atom for coset
            S_lZZ = {} # Z = (a, P)
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

            setup = setups[coset[0]]

            work = []
            for j, l in enumerate(setup.l_j):
                indices = []
                D_Z = np.zeros((nA * nP[l],))
                Z = 0
                for loca1, a1 in enumerate(coset):
                    a1_, start, end = layout.myindices[a1]
                    assert a1_ == a1
                    Itot = sum([2 * l2 + 1 for l2 in setup.l_j])
                    Istart = sum([2 * l2 + 1 for l2 in setup.l_j[:j]])
                    Iend = Istart + 2 * l + 1
                    for X in range(Istart, Iend):
                        for Y in range(Istart, Iend):
                            indices.append(start + X * Itot + Y)
                work.append((a, l, np.array(indices)))

        self.work = work
        self.S_alZZ = S_alZZ

    def apply(self, data):
        for a, l, indices in self.work:
            for spin in range(len(data)):
                print(a, l, indices)
                data[spin, indices] = self.S_alZZ[a][l] @ data[spin, indices]


if __name__ == "__main__":
    from ase.build import bulk
    from gpaw import GPAW
    #atoms = bulk('NaCl', 'rocksalt', a=4.0)
    atoms = bulk('Si')
    atoms *= 2
    calc = GPAW(mode='lcao', basis='sz(dzp)')
    atoms.calc = calc
    atoms.get_potential_energy()

    symmetries = calc.calculation.state.ibzwfs.ibz.symmetries

    rotation_lsmm = symmetries.rotation_lsmm
    a_sa = symmetries.a_sa
    D_asii = calc.calculation.state.density.D_asii
    
    symplan = SymmetrizationPlan(rotation_lsmm, a_sa, calc.setups, D_asii.layout)

    D_asii_data_before = D_asii.data.copy()
    symplan.apply(D_asii.data)
    D_asii_data_after = D_asii.data

    assert np.allclose(D_asii_data_before, D_asii_data_after)
    asd


    ns = a_sa.shape[0]
    na = a_sa.shape[1]

    cosets = {frozenset(a_sa[:, a]) for a in range(na)}
    print(cosets)
    nP = [(2 * l + 1)**2 for l in range(4)]

    S_lsPP = {l: np.einsum('sab,scd->sacbd', rotation_lsmm[l], rotation_lsmm[l]).reshape((ns, nP[l], nP[l])) / ns for l in range(4)}
    #for l in range(4):
    #    print('a',l, sorted(np.linalg.eig(np.sum(S_lsPP[l], axis=0))[0]))
    S_alZZ = {}
    D_asii_data_before = D_asii.data.copy()
    breakpoint()
    for coset in map(list, cosets):
        # Calculate general rotation matrices for orbits of particular angular momentum

        nA = len(coset)
        a = coset[0] # Representative atom for coset
        S_lZZ = {} # Z = (a, P)
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

        setup = calc.setups[coset[0]]

        for j, l in enumerate(setup.l_j):
            D_Z = np.zeros((nA * nP[l],))
            Z = 0
            for loca1, a1 in enumerate(coset):
                D_ii = D_asii[a1][0]
                Istart = sum([2 * l2 + 1 for l2 in setup.l_j[:j]])
                Iend = Istart + 2 * l + 1

                D_Z[Z:Z+nP[l]] = D_ii[Istart:Iend, Istart:Iend].flatten()
                Z += nP[l]
            
            D_Z = S_alZZ[a][l] @ D_Z # Actual work

            Z = 0
            for loca1, a1 in enumerate(coset):
                D_ii = D_asii[a1][0]
                Istart = sum([2 * l2 + 1 for l2 in setup.l_j[:j]])
                Iend = Istart + 2 * l + 1

                #print(f'Before {Istart}:{Iend} j={j} l={l} a={a} a1={a1}', D_ii[Istart:Iend, Istart:Iend])
                D_ii[Istart:Iend, Istart:Iend] = D_Z[Z:Z+nP[l]].reshape((2 * l + 1,) * 2)
                #print('After', D_ii[Istart:Iend, Istart:Iend])
                Z += nP[l]

                
    D_asii_data_after = D_asii.data
    for i, (b, a) in enumerate(zip(D_asii_data_before.ravel(), D_asii_data_after.ravel())):
        if np.abs(a-b)>1e-5:
            print(a,b, a-b, a/b)
    assert np.allclose(D_asii_data_after, D_asii_data_before)
