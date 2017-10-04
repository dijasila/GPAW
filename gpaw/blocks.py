import numpy as np

from gpaw.utilities import pack2, unpack


class AtomicBlocks:
    def __init__(self, M_asii, nspins=None, comm=None, size_a=None):
        self.M_asii = M_asii
        self.nspins = nspins
        self.comm = comm
        self.size_a = size_a

        # self.indices = []
        # I1 = 0
        # for a, M_sii in sorted(M_asii.items()):
        #     I2 = I1 + M_sii.shape[-1]
        #     self.indices.append((a, I1, I2))
        #     I1 = I2

        self.rank_a = None

    def __getitem__(self, a):
        return self.M_asii[a]

    def broadcast(self):
        M_asii = []
        for a, ni in enumerate(self.size_a):
            M_sii = self.M_asii.get(a)
            if M_sii is None:
                M_sii = np.empty((self.nspins, ni, ni))
            self.comm.broadcast(M_sii, self.rank_a[a])
            M_asii.append(M_sii)
        return M_asii

    def pack(self):
        M_asii = self.broadcast()
        P = sum(ni * (ni + 1) // 2 for ni in self.size_a)
        M_sP = np.empty((self.nspins, P))
        P1 = 0
        for ni, M_sii in zip(self.size_a, M_asii):
            P2 = P1 + ni * (ni + 1) // 2
            M_sP[:, P1:P2] = [pack2(M_ii) for M_ii in M_sii]
            P1 = P2
        return M_sP

    def unpack(self, M_sP):
        self.M_asii.clear()
        if M_sP is None:
            return
        P1 = 0
        for a, ni in enumerate(self.size_a):
            P2 = P1 + ni * (ni + 1) // 2
            self.M_asii[a] = np.array([unpack(M_p) for M_p in M_sP[:, P1:P2]])
            P1 = P2


class OverlapCorrections:
    def __init__(self, setups):
        self.setups = setups

    def apply(self, P, out):
        for a, I1, I2 in P.indices:
            dS_ii = self.setups[a].dO_ii
            out.matrix.array[:, I1:I2] = np.dot(P.matrix.array[:, I1:I2],
                                                dS_ii)
