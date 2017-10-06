import numpy as np

from gpaw.matrix import Matrix
from gpaw.mpi import serial_comm


class Projections:
    def __init__(self, nbands, nproj_a, acomm, bcomm, rank_a,
                 collinear=True, spin=0, dtype=float):
        self.nproj_a = np.asarray(nproj_a)
        self.acomm = acomm
        self.bcomm = bcomm
        self.collinear = collinear
        self.spin = spin

        self.rank_a = rank_a
        self.indices = []
        self.my_atom_indices = []
        self.map = {}
        I1 = 0
        for a, ni in enumerate(nproj_a):
            if acomm.rank == rank_a[a]:
                self.my_atom_indices.append(a)
                I2 = I1 + ni
                self.indices.append((a, I1, I2))
                I1 = I2
                self.map[a] = (I1, I2)

        self.matrix = Matrix(nbands, I1, dtype, dist=(bcomm, bcomm.size, 1))

    def new(self, bcomm='inherit', nbands=None, rank_a=None):
        if bcomm == 'inherit':
            bcomm = self.bcomm
        elif bcomm is None:
            bcomm = serial_comm
        return Projections(
            nbands or self.matrix.shape[0], self.nproj_a,
            self.acomm, bcomm,
            self.rank_a if rank_a is None else rank_a,
            self.collinear, self.spin, self.matrix.dtype)

    def items(self):
        for a, I1, I2 in self.indices:
            yield a, self.matrix.array[:, I1:I2]

    def __getitemmmmmmmm__(self, a):
        I1, I2 = self.map[a]
        return self.matrix.array[I1:I2]

    def __contains__(self, a):
        return a in self.map

    def todicttttt(self):
        return dict(self.items())

    def redist(self, rank_a):
        P = self.new(rank_a=rank_a)
        P_In = self.collect_atoms(self.matrix)
        if self.acomm.rank == 0:
            mynbands = P_In.shape[1]
            for rank in range(self.acomm.size):
                nI = self.nproj_a[rank_a == rank].sum()
                if nI == 0:
                    continue
                P2_nI = np.empty((mynbands, nI), P_In.dtype)
                I1 = 0
                myI1 = 0
                for a, ni in enumerate(self.nproj_a):
                    I2 = I1 + ni
                    if rank == rank_a[a]:
                        myI2 = myI1 + ni
                        P2_nI[:, myI1:myI2] = P_In[I1:I2].T
                        myI1 = myI2
                    I1 = I2
                if rank == 0:
                    P.matrix.array[:] = P2_nI
                else:
                    self.acomm.send(P2_nI, rank)
        else:
            if P.matrix.array.size > 0:
                self.acomm.receive(P.matrix.array, 0)
        return P

    def collect(self):
        if self.bcomm.size == 1:
            P = self.matrix
        else:
            P = self.matrix.new(dist=(self.bcomm, 1, 1))
            self.matrix.redist(P)

        if self.bcomm.rank > 0:
            return None

        if self.acomm.size == 1:
            return P.array

        P_In = self.collect_atoms(P)
        if P_In is not None:
            return P_In.T

    def collect_atoms(self, P):
        if self.acomm.rank == 0:
            nproj = sum(self.nproj_a)
            P_In = np.empty((nproj, P.array.shape[0]), dtype=P.array.dtype)

            I1 = 0
            myI1 = 0
            for nproj, rank in zip(self.nproj_a, self.rank_a):
                I2 = I1 + nproj
                if rank == 0:
                    myI2 = myI1 + nproj
                    P_In[I1:I2] = P.array[:, myI1:myI2].T
                    myI1 = myI2
                else:
                    self.acomm.receive(P_In[I1:I2], rank)
                I1 = I2
            return P_In
        else:
            for a, I1, I2 in self.indices:
                self.acomm.send(P.array[:, I1:I2].T.copy(), 0)
            return None
