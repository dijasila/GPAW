import numpy as np

from gpaw.utilities.blas import gemm
from gpaw.utilities import unpack


def get_atomic_hamiltonian(name):
    cls = dict(dense=DenseAtomicHamiltonian,
               distributed=DistributedAtomicHamiltonian,
               scipy=ScipyAtomicHamiltonian)[name]
    return cls()


class DenseAtomicHamiltonian:
    name = 'dense'
    description = 'dense with blas'
    nops = 0

    def gobble_data(self, wfs):
        pass # No complex data structures

    def redistribute(self, wfs, dH_asp):
        return dH_asp
    
    def calculate(self, wfs, kpt, dH_asp, H_MM, yy):
        Mstart = wfs.ksl.Mstart
        Mstop = wfs.ksl.Mstop
        dtype = wfs.dtype
        nops = 0
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(dH_asp[a][kpt.s]), dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            nops += dHP_iM.size * dH_ii.shape[0]
            gemm(yy, dHP_iM, P_Mi[Mstart:Mstop], 1.0, H_MM)
            nops += H_MM.size * dHP_iM.shape[0]
        self.nops = nops


class DistributedAtomicHamiltonian:
    name = 'distributed'
    description = 'distributed and block-sparse'
    nops = 0

    def gobble_data(self, wfs):
        pass # XXX Move some preparation stuff here
    
    def redistribute(self, wfs, dH_asp):
        def get_empty(a):
            ni = wfs.setups[a].ni
            return np.empty((wfs.ns, ni * (ni + 1) // 2))

        # just distributed over gd comm.  It's not the most aggressive
        # we can manage but we want to make band parallelization
        # a bit easier and it won't really be a problem.  I guess
        #
        # Also: This call is blocking, but we could easily do a
        # non-blocking version as we only need this stuff after
        # doing tons of real-space work.

        return wfs.atom_partition.to_even_distribution(dH_asp,
                                                       get_empty,
                                                       copy=True)

    def calculate(self, wfs, kpt, dH_asp, H_MM, yy):
        # XXX reduce according to kpt.q
        dtype = wfs.dtype
        M_a = wfs.setups.M_a
        nM_a = np.array([setup.nao for setup in wfs.setups])
        Mstart = wfs.ksl.Mstart
        Mstop = wfs.ksl.Mstop

        # Now calculate basis-projector-basis overlap: a1 -> a3 -> a2
        #
        # specifically:
        #   < phi[a1] | p[a3] > * dH[a3] * < p[a3] | phi[a2] >
        #
        # This matrix multiplication is semi-sparse.  It works by blocks
        # of atoms, looping only over pairs that do have nonzero
        # overlaps.  But it might be even nicer with scipy sparse.
        # This we will have to check at some point.
        #
        # The projection arrays P_aaim are distributed over the grid,
        # whereas the Hamiltonian is distributed over the band comm.
        # One could choose a set of a3 to optimize the load balance.
        # Right now the load balance will be "random" and probably
        # not very good.
        #innerloops = 0

        setups = wfs.setups

        outer = 0
        inner = 0
        nops = 0

        nao_a = [setup.nao for setup in setups]

        for (a3, a1), P1_im in kpt.P_aaim.items():
            a1M1 = M_a[a1]
            nM1 = nM_a[a1]
            a1M2 = a1M1 + nM1

            if a1M1 > Mstop or a1M2 < Mstart:
                continue

            stickout1 = max(0, Mstart - a1M1)
            stickout2 = max(0, a1M2 - Mstop)
            P1_mi = np.conj(P1_im.T[stickout1:nM1 - stickout2])
            dH_ii = yy * np.asarray(unpack(dH_asp[a3][kpt.s]), dtype)
            H_mM = H_MM[a1M1 + stickout1 - Mstart:a1M2 - stickout2 - Mstart]

            P1dH_mi = np.dot(P1_mi, dH_ii)
            nops += P1dH_mi.size * dH_ii.shape[0]
            outer += 1

            assert len(wfs.P_neighbors_a[a3]) > 0
            if 0:
                for a2 in wfs.P_neighbors_a[a3]:
                    inner += 1
                    # We can use symmetry somehow.  Since the entire matrix
                    # is symmetrized after the Hamiltonian is constructed,
                    # at least in the non-Gamma-point case, we should do
                    # so conditionally somehow.  Right now let's stay out
                    # of trouble.

                    # Humm.  The following works with gamma point
                    # but not with kpts.  XXX take a look at this.
                    # also, it doesn't work with a2 < a1 for some reason.
                    #if a2 > a1:
                    #    continue
                    a2M1 = M_a[a2]
                    a2M2 = a2M1 + wfs.setups[a2].nao
                    P2_im = kpt.P_aaim[(a3, a2)]
                    P1dHP2_mm = np.dot(P1dH_mi, P2_im)
                    H_mM[:, a2M1:a2M2] += P1dHP2_mm
                    #innerloops += 1
                    #if wfs.world.rank == 0:
                    #    print 'y', y
                    #if a1 != a2:
                    #    H_MM[a2M1:a2M2, a1M1:a1M2] += P1dHP2_mm.T.conj()

            a2_a = wfs.P_neighbors_a[a3]

            a2nao = sum(nao_a[a2] for a2 in a2_a)

            P2_iam = np.empty((P1_mi.shape[1], a2nao), dtype)
            m2 = 0
            for a2 in a2_a:
                P2_im = kpt.P_aaim[(a3, a2)]
                nao = nao_a[a2]
                P2_iam[:, m2:m2 + nao] = P2_im
                m2 += nao

            P1dHP2_mam = np.zeros((P1dH_mi.shape[0],
                                   P2_iam.shape[1]), dtype)
            gemm(1.0, P2_iam, P1dH_mi, 0.0, P1dHP2_mam)
            nops += P1dHP2_mam.size * P2_iam.shape[0]

            m2 = 0
            for a2 in a2_a:
                nao = nao_a[a2]
                H_mM[:, M_a[a2]:M_a[a2] + setups[a2].nao] += \
                    P1dHP2_mam[:, m2:m2 + nao]
                m2 += nao

        self.nops = nops
        self.inner = inner
        self.outer = outer

class ScipyAtomicHamiltonian(DistributedAtomicHamiltonian):
    name = 'scipy'
    description = 'distributed and sparse using scipy'
    nops = 0

    def __init__(self):
        from scipy import sparse
        self.sparse = sparse

    def gobble_data(self, wfs):
        nq = len(wfs.kd.ibzk_qc)

        I_a = [0]
        I_a.extend(np.cumsum([setup.ni for setup in wfs.setups[:-1]]))
        I = I_a[-1] + wfs.setups[-1].ni
        self.I = I
        self.I_a = I_a

        M_a = wfs.setups.M_a
        M = M_a[-1] + wfs.setups[-1].nao
        nao_a = [setup.nao for setup in wfs.setups]
        ni_a = [setup.ni for setup in wfs.setups]

        Psparse_qIM = [self.sparse.lil_matrix((I, M), dtype=wfs.dtype)
                       for _ in range(nq)]

        for (a3, a1), P_qim in wfs.P_aaqim.items():
            P_qim = P_qim.copy()
            approximately_zero = np.abs(P_qim) < 1e-12
            P_qim[approximately_zero] = 0.0
            for q in range(nq):
                Psparse_qIM[q][I_a[a3]:I_a[a3] + ni_a[a3],
                               M_a[a1]:M_a[a1] + nao_a[a1]] = P_qim[q]
        
        self.Psparse_qIM = [x.tocsr() for x in Psparse_qIM]


    def calculate(self, wfs, kpt, dH_asp, H_MM, yy):
        Mstart = wfs.ksl.Mstart
        Mstop = wfs.ksl.Mstop

        aval = dH_asp.keys()
        aval.sort()
        Psparse_IM = self.Psparse_qIM[kpt.q]

        dHsparse_II = self.sparse.lil_matrix((self.I, self.I), dtype=wfs.dtype)
        for a in aval:
            I1 = self.I_a[a]
            I2 = I1 + wfs.setups[a].ni
            dHsparse_II[I1:I2, I1:I2] = yy * unpack(dH_asp[a][kpt.s])
        dHsparse_II = dHsparse_II.tocsr()
        
        Psparse_MI = Psparse_IM[:, Mstart:Mstop].transpose().conjugate()
        Hsparse_MM = Psparse_MI.dot(dHsparse_II.dot(Psparse_IM))
        H_MM[:, :] += Hsparse_MM.todense()
