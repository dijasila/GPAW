import numpy as np

from gpaw.matrix import Matrix
from gpaw.mpi import serial_comm


class Projections:
    def __init__(self, nproj_a, nbands, acomm, bcomm, rank_a,
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

        self.matrix = Matrix(I1, nbands, dtype, dist=(bcomm, 1, bcomm.size),
                             order='F')

    def new(self, bcomm='inherit', nbands=None, rank_a=None):
        if bcomm == 'inherit':
            bcomm = self.bcomm
        elif bcomm is None:
            bcomm = serial_comm
        return Projections(
            self.nproj_a, nbands or self.matrix.shape[1],
            self.acomm, bcomm,
            self.rank_a if rank_a is None else rank_a,
            self.collinear, self.spin, self.matrix.dtype)

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

    def redist(self, rank_a):
        P = self.new(rank_a=rank_a)
        P_In = self.collect_atoms(self.matrix)
        if self.acomm.rank == 0:
            mynbands = P_In.shape[1]
            for rank in range(self.acomm.size):
                nI = self.nproj_a[rank_a == rank].sum()
                if nI == 0:
                    continue
                P2_In = np.empty((nI, mynbands), P_In.dtype, order='F')
                I1 = 0
                myI1 = 0
                for a, ni in enumerate(self.nproj_a):
                    I2 = I1 + ni
                    if rank == rank_a[a]:
                        myI2 = myI1 + ni
                        P2_In[myI1:myI2] = P_In[I1:I2]
                        myI1 = myI2
                    I1 = I2
                if rank == 0:
                    P.matrix.array[:] = P2_In
                else:
                    self.acomm.send(P2_In.T, rank)
        else:
            if P.matrix.array.size > 0:
                self.acomm.receive(P.matrix.array.T, 0)
        return P

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

        return self.collect_atoms(P)

    def collect_atoms(self, P):
        if self.acomm.rank == 0:
            nproj = sum(self.nproj_a)
            P_In = np.empty((nproj, P.array.shape[1]),
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
                    P_in = P_In[I1:I2]
                    tmp_in = np.empty(P_in.shape, dtype=P_in.dtype)
                    self.acomm.receive(tmp_in, rank)
                    P_in[:] = tmp_in
                I1 = I2
            return P_In
        else:
            for a, I1, I2 in self.indices:
                self.acomm.send(P.array[I1:I2].copy(), 0)
            return None
