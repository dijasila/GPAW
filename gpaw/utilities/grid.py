from __future__ import print_function
from functools import partial

import numpy as np
from gpaw.utilities.grid_redistribute import general_redistribute


class Grid2Grid:
    def __init__(self, comm, broadcast_comm, gd, big_gd, enabled):
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
    
    def distribute(self, src_g, dst_g):
        self._distribute(src_g, dst_g)
    
    def collect(self, src_g, dst_g):
        self._collect(src_g, dst_g)
        self.broadcast_comm.broadcast(dst_g, 0)

    # Strangely enough the purpose of this is to appease AtomPAW
    def new(self, gd, big_gd):
        return Grid2Grid(self.comm, self.broadcast_comm, gd, big_gd,
                         self.enabled)


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
