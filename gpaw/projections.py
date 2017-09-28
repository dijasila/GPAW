import numpy as np

from gpaw.matrix import Matrix
from gpaw.mpi import serial_comm


class Projections:
    def __init__(self, nproj_a, nbands, acomm, bcomm, rank_a,
                 collinear=True, spin=0, dtype=float):
        self.nproj_a = nproj_a
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

        self.matrix = Matrix(I1, nbands, dtype, dist=(bcomm, 1, bcomm.size),
                             order='F')

    def new(self, bcomm='inherit', nbands=None):
        if bcomm == 'inherit':
            bcomm = self.bcomm
        elif bcomm is None:
            bcomm = serial_comm
        return Projections(
            self.nproj_a, nbands or self.matrix.shape[1],
            self.acomm, bcomm,
            self.rank_a, self.collinear, self.spin, self.matrix.dtype)

    def items(self):
        for a, I1, I2 in self.indices:
            yield a, self.matrix.array[I1:I2]

    def __getitemmmmmmmm__(self, a):
        I1, I2 = self.map[a]
        return self.matrix.array[I1:I2]

    def __contains__(self, a):
        return a in self.map

    def todicttttt(self):
        return dict(self.items())

    def collect(self):
        if self.bcomm.size == 1:
            P = self.matrix
        else:
            comm = self.bcomm.new_communicator([0])
            P = self.matrix.new(dist=comm)
            self.matrix.redist(P)

        if self.bcomm.rank > 0:
            return None

        if self.acomm.size == 1:
            return P.array

        if self.acomm.rank == 0:
            nproj = sum(self.nproj_a)
            P_In = np.empty((nproj, P.shape[1]),
                            dtype=P.array.dtype, order='F')

            I1 = 0
            myI1 = 0
            for nproj, rank in zip(self.nproj_a, self.rank_a):
                I2 = I1 + nproj
                if rank == 0:
                    myI2 = myI1 + nproj
                    P_In[I1:I2] = P.array[myI1:myI2]
                    myI1 = myI2
                else:
                    self.acomm.receive(P_In[I1:I2], rank)
                I1 = I2
            return P_In
        else:
            for a, I1, I2 in self.indices:
                self.acomm.send(P.array[I1:I2].copy(), 0)
            return None
