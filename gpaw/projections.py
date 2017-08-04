import numpy as np

from gpaw.matrix import Matrix


class Projections:
    def __init__(self, nproj_a, nbands, acomm, bcomm, rank_a,
                 collinear=True, spin=0, dtype=float):
        self.nproj_a = nproj_a
        self.acomm = acomm
        self.bcomm = bcomm
        self.rank_a = rank_a
        self.collinear = collinear
        self.spin = spin

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

        self.matrix = Matrix(I1, nbands, dtype, dist=(bcomm, 1, -1))

    def new(self):
        return Projections(
            self.nproj_a, self.shape[1], self.acomm, self.bcomm,
            self.rank_a, self.collinear, self.spin, self.dtype)

    def items(self):
        for a, I1, I2 in self.indices:
            yield a, self.matrix.array[I1:I2]

    def __getitem__(self, a):
        I1, I2 = self.map[a]
        return self.matrix.array[I1:I2]

    def __contains__(self, a):
        return a in self.map

    def todict(self):
        return dict(self.items())

    def collect(self):
        assert self.acomm.size == 1
        assert self.bcomm.size == 1
        return self.matrix.array
        """
        natoms = self.atom_partition.natoms

        if self.world.rank == 0:
            if kpt_rank == 0:
                P_ani = self.kpt_u[u].P_ani
            all_P_ni = np.empty((self.bd.nbands, nproj), self.dtype)
            for band_rank in range(self.bd.comm.size):
                nslice = self.bd.get_slice(band_rank)
                i = 0
                for a in range(natoms):
                    ni = self.setups[a].ni
                    if kpt_rank == 0 and band_rank == 0 and a in P_ani:
                        P_ni = P_ani[a]
                    else:
                        P_ni = np.empty((self.bd.mynbands, ni), self.dtype)
                        # XXX will fail with nonstandard communicator nesting
                        world_rank = (self.atom_partition.rank_a[a] +
                                      kpt_rank * self.gd.comm.size *
                                      self.bd.comm.size +
                                      band_rank * self.gd.comm.size)
                        self.world.receive(P_ni, world_rank, 1303 + a)
                    all_P_ni[nslice, i:i + ni] = P_ni
                    i += ni
                assert i == nproj

            if asdict:
                i = 0
                P_ani = {}
                for a in range(natoms):
                    ni = self.setups[a].ni
                    P_ani[a] = all_P_ni[:, i:i + ni]
                    i += ni
                return P_ani

            return all_P_ni
        """
