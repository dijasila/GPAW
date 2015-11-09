from __future__ import print_function
from functools import partial

import numpy as np
from gpaw.utilities.grid_redistribute import general_redistribute
from gpaw.utilities.partition import AtomPartition, AtomicMatrixDistributor


class Grid2Grid:
    def __init__(self, comm, broadcast_comm, gd, big_gd, enabled=True):
        self.comm = comm
        self.broadcast_comm = broadcast_comm
        self.gd = gd
        self.big_gd = big_gd
        self.enabled = enabled
        
        if big_gd.comm.rank == 0:
            big_ranks = gd.comm.translate_ranks(big_gd.comm,
                                                np.arange(gd.comm.size))
        else:
            big_ranks = np.empty(gd.comm.size, dtype=int)
        big_gd.comm.broadcast(big_ranks, 0)
        
        bigrank2rank = dict(zip(big_ranks, np.arange(gd.comm.size)))
        def rank2parpos1(rank):
            if rank in bigrank2rank:
                return gd.get_processor_position_from_rank(bigrank2rank[rank])
            else:
                return None

        rank2parpos2 = big_gd.get_processor_position_from_rank

        self._distribute = partial(general_redistribute, big_gd.comm,
                                   gd.n_cp, big_gd.n_cp,
                                   rank2parpos1, rank2parpos2)
        self._collect = partial(general_redistribute, big_gd.comm,
                                big_gd.n_cp, gd.n_cp,
                                rank2parpos2, rank2parpos1)
    
    def distribute(self, src_xg, dst_xg=None):
        if dst_xg is None:
            dst_xg = self.big_gd.empty(src_xg.shape[:-3], dtype=src_xg.dtype)
        self._distribute(src_xg, dst_xg)
        return dst_xg
    
    def collect(self, src_xg, dst_xg=None):
        if dst_xg is None:
            dst_xg = self.gd.empty(src_xg.shape[:-3], src_xg.dtype)
        self._collect(src_xg, dst_xg)
        self.broadcast_comm.broadcast(dst_xg, 0)
        return dst_xg

    # Strangely enough the purpose of this is to appease AtomPAW
    def new(self, gd, big_gd):
        return Grid2Grid(self.comm, self.broadcast_comm, gd, big_gd,
                         self.enabled)

    def get_matrix_distributor(self, atom_partition, spos_ac=None):
        if spos_ac is None:
            rank_a = np.zeros(self.big_gd.comm.size, dtype=int)
        else:
            rank_a = self.big_gd.get_ranks_from_positions(spos_ac)
        big_partition = AtomPartition(self.big_gd.comm, rank_a)
        return AtomicMatrixDistributor(atom_partition, self.broadcast_comm,
                                       big_partition)


class NullGrid2Grid:
    def __init__(self, aux_gd):
        self.gd = aux_gd
        self.big_gd = aux_gd
        self.enabled = False

    def distribute(self, src_xg, dst_xg=None):
        assert src_xg is dst_xg or dst_xg is None
        return src_xg

    collect = distribute

    def get_matrix_distributor(self, atom_partition, spos_ac=None):
        class NullMatrixDistributor:
            def distribute(self, D_asp):
                return D_asp
            collect = distribute
        return NullMatrixDistributor()

    def new(self, gd, big_gd):
        assert np.all(gd.n_c == big_gd.n_c)
        return NullGrid2Grid(gd)


def grid2grid(comm, gd1, gd2, src_g, dst_g):
    assert np.all(src_g.shape == gd1.n_c)
    assert np.all(dst_g.shape == gd2.n_c)

    #master1_rank = gd1.comm.translate_ranks(comm, [0])[0]
    #master2_rank = gd2.comm.translate_ranks(comm, [0])[0]

    ranks1 = gd1.comm.translate_ranks(comm, np.arange(gd1.comm.size))
    ranks2 = gd2.comm.translate_ranks(comm, np.arange(gd2.comm.size))
    assert (ranks1 >= 0).all(), 'comm not parent of gd1.comm'
    assert (ranks2 >= 0).all(), 'comm not parent of gd2.comm'

    def rank2parpos(gd, rank):
        gdrank = comm.translate_ranks(gd.comm, [rank])[0]
        if gdrank == -1:
            return None
        return gd.get_processor_position_from_rank(gdrank)
    rank2parpos1 = partial(rank2parpos, gd1)
    rank2parpos2 = partial(rank2parpos, gd2)

    general_redistribute(comm,
                         gd1.n_cp, gd2.n_cp,
                         rank2parpos1, rank2parpos2,
                         src_g, dst_g)

def main():
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import world
    
    serial = world.new_communicator([world.rank])

    # Genrator which must run on all ranks
    gen = np.random.RandomState(0)

    # This one is just used by master
    gen_serial = np.random.RandomState(17)

    maxsize = 5
    for i in range(1):
        N1_c = gen.randint(1, maxsize, 3)
        N2_c = gen.randint(1, maxsize, 3)
        
        gd1 = GridDescriptor(N1_c, N1_c)
        gd2 = GridDescriptor(N2_c, N2_c)
        serial_gd1 = gd1.new_descriptor(comm=serial)
        serial_gd2 = gd2.new_descriptor(comm=serial)

        a1_serial = serial_gd1.empty()
        a1_serial.flat[:] = gen_serial.rand(a1_serial.size)

        if world.rank == 0:
            print('r0: a1 serial', a1_serial.ravel())

        a1 = gd1.empty()
        a1[:] = -1

        grid2grid(world, serial_gd1, gd1, a1_serial, a1)

        print(world.rank, 'a1 distributed', a1.ravel())
        world.barrier()

        a2 = gd2.zeros()
        a2[:] = -2
        grid2grid(world, gd1, gd2, a1, a2)
        print(world.rank, 'a2 distributed', a2.ravel())
        world.barrier()

        #grid2grid(world, gd2, gd2_serial

        gd1 = GridDescriptor(N1_c, N1_c * 0.2)
        #serialgd = gd2.new_descriptor(

        a1 = gd1.empty()
        a1.flat[:] = gen.rand(a1.size)

        #print a1
        grid2grid(world, gd1, gd2, a1, a2)

        #print a2

if __name__ == '__main__':
    main()
