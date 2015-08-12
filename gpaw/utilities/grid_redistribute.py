# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from gpaw.grid_descriptor import GridDescriptor


class GridRedistributor:
    """Perform redistributions between two grids.

    See the redistribute function."""
    def __init__(self, gd, distribute_dir, reduce_dir):
        self.gd = gd
        self.distribute_dir = distribute_dir
        self.reduce_dir = reduce_dir
        self.gd2 = get_compatible_grid_descriptor(gd, distribute_dir,
                                                  reduce_dir)

    def _redist(self, src, op):
        return redistribute(self.gd, self.gd2, src, self.distribute_dir,
                            self.reduce_dir, operation=op)
        
    def forth(self, src):
        return self._redist(src, 'forth')

    def back(self, src):
        return self._redist(src, 'back')


def redistribute(gd, gd2, src, distribute_dir, reduce_dir, operation='forth'):
    """Perform certain simple redistributions among two grid descriptors.

    Redistribute src from gd with decomposition X x Y x Z to gd2 with
    decomposition X x YZ x 1, or some variation of this.  We say that
    we "reduce" along Z while we "distribute" along Y.  The
    redistribution is one-to-one.

             ____________                           ____________
    i       /     /     /|          r              /  /  /  /  /|
    n      /_____/_____/ |         i              /  /  /  /  / |
    d     /     /     /| |        d              /  /  /  /  /  |
    e    /_____/_____/ | j           forth      /__/__/__/__/   j
    p    |     |     | |/|      e    ------->   |  |  |  |  |  /|
    e    |     |     | ł |     c    <-------    |  |  |  |  | / |
    n    |_____|_____|/| j    u       back      |__|__|__|__|/  j
    d    |     |     | |/    d                  |  |  |  |  |  /
    e    |     |     | /    e                   |  |  |  |  | /
    n    |_____|_____|/    r                    |__|__|__|__|/
    t

         d i s t r i b u t e   d i r

    Directions are specified as 0, 1, or 2.  gd2 must be serial along
    the axis of reduction and must parallelize enough over the
    distribution axis to match the size of gd.comm.

    Returns the redistributed array which is compatible with gd2.

    Note: The communicator of gd2 must in general be a special
    permutation of that of gd in order for the redistribution axes to
    align with domain rank assignment.  Use the helper function
    get_compatible_grid_descriptor to obtain a grid descriptor which
    uses a compatible communicator."""

    assert reduce_dir != distribute_dir
    assert gd.comm.size == gd2.comm.size
    # Actually: The two communicators should be equal!!
    for c in [reduce_dir, distribute_dir]:
        assert 0 <= c and c < 3

    # Determine the direction in which nothing happens.
    for c in range(3):
        if c != reduce_dir and c != distribute_dir:
            independent_dir = c
            break
    assert np.all(gd.N_c == gd2.N_c)
    assert np.all(gd.pbc_c == gd2.pbc_c)
    assert gd.n_c[independent_dir] == gd2.n_c[independent_dir]
    assert gd.parsize_c[independent_dir] == gd2.parsize_c[independent_dir]
    assert gd2.parsize_c[reduce_dir] == 1
    assert gd2.parsize_c[distribute_dir] == gd.parsize_c[reduce_dir] \
        * gd.parsize_c[distribute_dir]
    assert operation == 'forth' or operation == 'back'
    forward = (operation == 'forth')
    if forward:
        assert np.all(src.shape == gd.n_c)
    else:
        assert np.all(src.shape == gd2.n_c)
    assert gd.comm.compare(gd2.comm) != 'unequal'

    # We want this to work no matter which direction is distribute and
    # reduce.  But that is tricky to code.  So we use a standard order
    # of the three directions.
    #
    # Thus we have to always transpose the src/dst arrays consistently
    # when interacting with the contiguous MPI send/recv buffers.  An
    # alternative is to use np.take, but that sometimes produces
    # copies where slicing does not, and we want to write back into
    # slices.
    #
    # We only support some of them though...
    dirs = (independent_dir, distribute_dir, reduce_dir)
    src = src.transpose(*dirs)

    # Construct a communicator consisting of all those processes that
    # participate in domain decomposition along the reduction
    # direction.
    #
    # All necessary communication can be done within that
    # subcommunicator using MPI alltoallv.
    #
    # We also construct the "same" communicator from gd2.comm, but with the
    # sole purpose of testing that the ranks are consistent between the two.
    # If they are not, the two grid descriptors are incompatible.
    pos_c = gd.parpos_c.copy()
    pos2_c = gd2.parpos_c.copy()
    positions2_offset = pos_c[distribute_dir] * gd.parsize_c[reduce_dir]
    peer_ranks = []
    peer_ranks2 = []
    for i in range(gd.parsize_c[reduce_dir]):
        pos_c[reduce_dir] = i
        pos2_c[distribute_dir] = i + positions2_offset
        peer_ranks.append(gd.get_rank_from_processor_position(pos_c))
        peer_ranks2.append(gd2.get_rank_from_processor_position(pos2_c))
    peer_comm = gd.comm.new_communicator(peer_ranks)
    test_peer_comm2 = gd2.comm.new_communicator(peer_ranks2)
    if test_peer_comm2.compare(peer_comm) != 'congruent':
        raise ValueError('Grids are not compatible.  '
                         'Use get_compatible_grid_descriptor to construct '
                         'a compatible grid.')
    #assert peer_comm2 is not None
    assert peer_comm.compare(gd2.comm.new_communicator(peer_ranks2)) == 'congruent'
    #print('COMPARE', peer_ranks, peer_ranks2, peer_comm.compare(peer_comm2))

    # Now check that peer_comm encompasses the same physical processes
    # on the communicators of the two grid descriptors.
    #test1 = peer_comm.translate_ranks(gd.comm, np.arange(peer_comm.size))
    #test2 = peer_comm.translate_ranks(gd.comm, np.arange(peer_comm.size))
    #print(peer_comm)

    members = peer_comm.get_members()

    mynpts1_rdir = gd.n_c[reduce_dir]
    mynpts2_ddir = gd2.n_c[distribute_dir]
    mynpts_idir = gd.n_c[independent_dir]
    assert mynpts_idir == gd2.n_c[independent_dir]

    offsets1_rdir_p = gd.n_cp[reduce_dir]
    offsets2_ddir_p = gd2.n_cp[distribute_dir]

    npts1_rdir_p = offsets1_rdir_p[1:] - offsets1_rdir_p[:-1]
    npts2_ddir_p = offsets2_ddir_p[1:] - offsets2_ddir_p[:-1]

    # We have the sendbuffer, and it is contiguous.  But the parts
    # that we are going to send to each CPU are not contiguous!  We
    # need to loop over all the little chunks that we want to send,
    # and put them into a contiguous buffer for MPI.  Moreover, the
    # received data will unsurprisingly be in that very same order.
    # Therefore, we need to know how to unpack those data and put them
    # into the return array too.
    #
    # The following loop builds the send buffer, and manages the logic
    # for the receive buffer.  However since we have not received the
    # data, we obviously cannot copy anything out of the receive
    # buffer yet.  Therefore we create a list of ChunkCopiers that
    # contain all the information that they need to later copy things
    # into the appropriate places of the return array.

    if forward:
        dst = gd2.zeros(dtype=src.dtype)
    else:
        dst = gd.zeros(dtype=src.dtype)
    recvbuf = np.empty(dst.size, dtype=src.dtype)
    dst[:] = -2
    recvbuf[:] = -3

    sendchunks = []
    recvchunks = []
    recv_chunk_copiers = []
    
    class ChunkCopier:
        def __init__(self, src_chunk, dst_chunk):
            self.src_chunk = src_chunk
            self.dst_chunk = dst_chunk

        def copy(self):
            self.dst_chunk.flat[:] = self.src_chunk

    # Convert from peer_comm
    ranks1to2 = gd.comm.translate_ranks(gd2.comm, np.arange(gd.comm.size))
    assert (ranks1to2 != -1).all()

    recvchunk_start = 0
    for i in range(peer_comm.size):
        parent_rank = members[i]
        parent_rank2 = ranks1to2[parent_rank]

        parent_coord1 = \
            gd.get_processor_position_from_rank(parent_rank)[reduce_dir]
        parent_coord2 = \
            gd2.get_processor_position_from_rank(parent_rank2)[distribute_dir]

        # Warning: Many sendXXX and recvXXX variables are badly named
        # because they change roles when operation='back'.
        sendstart_ddir = offsets2_ddir_p[parent_coord2] \
            - gd.beg_c[distribute_dir]
        sendstop_ddir = sendstart_ddir + npts2_ddir_p[parent_coord2]
        sendnpts_ddir = sendstop_ddir - sendstart_ddir

        # Compensate for the infinitely annoying convention that enumeration
        # of points starts at 1 in non-periodic directions.
        #
        # Also, if we want to handle more general redistributions, the
        # below buffers must have something subtracted to get a proper
        # local index.
        recvstart_rdir = offsets1_rdir_p[parent_coord1] \
            - 1 + gd.pbc_c[reduce_dir]
        recvstop_rdir = recvstart_rdir + npts1_rdir_p[parent_coord1]
        recvnpts_rdir = recvstop_rdir - recvstart_rdir

        # Grab subarray that is going to be sent to process i.
        if forward:
            assert 0 <= sendstart_ddir
            assert sendstop_ddir <= src.shape[1]
            sendchunk = src[:, sendstart_ddir:sendstop_ddir, :]
            assert sendchunk.size == mynpts1_rdir * sendnpts_ddir * mynpts_idir, (sendchunk.shape, (mynpts_idir, sendnpts_ddir, mynpts1_rdir))
        else:
            sendchunk = src[:, :, recvstart_rdir:recvstop_rdir]
            assert sendchunk.size == recvnpts_rdir * mynpts2_ddir * mynpts_idir
        sendchunks.append(sendchunk)

        if forward:
            recvchunksize = recvnpts_rdir * mynpts2_ddir * mynpts_idir
        else:
            recvchunksize = mynpts1_rdir * sendnpts_ddir * mynpts_idir
        recvchunk = recvbuf[recvchunk_start:recvchunk_start + recvchunksize]
        recvchunks.append(recvchunk)
        recvchunk_start += recvchunksize

        if forward:
            dstchunk = dst.transpose(*dirs)[:, :, recvstart_rdir:recvstop_rdir]
        else:
            dstchunk = dst.transpose(*dirs)[:, sendstart_ddir:sendstop_ddir, :]
        copier = ChunkCopier(recvchunk, dstchunk)
        recv_chunk_copiers.append(copier)

    sendcounts = np.array([chunk.size for chunk in sendchunks], dtype=int)
    recvcounts = np.array([chunk.size for chunk in recvchunks], dtype=int)

    assert sendcounts.sum() == src.size
    assert recvcounts.sum() == dst.size
    senddispls = np.array([0] + list(np.cumsum(sendcounts))[:-1], dtype=int)
    recvdispls = np.array([0] + list(np.cumsum(recvcounts))[:-1], dtype=int)

    sendbuf = np.concatenate([sendchunk.ravel() for sendchunk in sendchunks])

    peer_comm.alltoallv(sendbuf, sendcounts, senddispls,
                        recvbuf, recvcounts, recvdispls)
        
    # Copy contiguous blocks of receive buffer back into precoded slices:
    for chunk_copier in recv_chunk_copiers:
        chunk_copier.copy()
    return dst


def get_compatible_grid_descriptor(gd, distribute_dir, reduce_dir):
    
    parsize2_c = list(gd.parsize_c)
    parsize2_c[reduce_dir] = 1
    parsize2_c[distribute_dir] = gd.parsize_c[reduce_dir] \
        * gd.parsize_c[distribute_dir]

    # Because of the way in which domains are assigned to ranks, some
    # redistributions cannot be represented on any grid descriptor
    # that uses the same communicator.  However we can create a
    # different one which assigns ranks in a manner corresponding to
    # a permutation of the axes, and there always exists a compatible
    # such communicator.

    # Probably there are two: a left-handed and a right-handed one
    # (i.e., positive or negative permutation of the axes).  It would
    # probably be logical to always choose a right-handed one.  Right
    # now the numbers correspond to whatever first was made to work
    # though!
    t = {(0, 1): (0, 1, 2),
         (0, 2): (0, 2, 1),
         (1, 0): (1, 0, 2),
         (1, 2): (0, 1, 2),
         (2, 1): (0, 2, 1),
         (2, 0): (1, 2, 0)}[(distribute_dir, reduce_dir)]
    
    ranks = np.arange(gd.comm.size).reshape(gd.parsize_c).transpose(*t).ravel()
    comm2 = gd.comm.new_communicator(ranks)
    gd2 = gd.new_descriptor(comm=comm2, parsize=parsize2_c)
    return gd2

def playground():
    np.set_printoptions(linewidth=176)
    #N_c = [4, 7, 9]
    N_c = [4, 4, 2]

    pbc_c = (1, 1, 1)

    # 210
    distribute_dir = 1
    reduce_dir = 0

    parsize_c = (2, 2, 2)
    parsize2_c = list(parsize_c)
    parsize2_c[reduce_dir] = 1
    parsize2_c[distribute_dir] *= parsize_c[reduce_dir]
    assert np.prod(parsize2_c) == np.prod(parsize_c)

    gd = GridDescriptor(N_c=N_c, pbc_c=pbc_c, cell_cv=0.2 * np.array(N_c),
                        parsize=parsize_c)

    gd2 = get_compatible_grid_descriptor(gd, distribute_dir, reduce_dir)

    src = gd.zeros(dtype=complex)
    src[:] = gd.comm.rank

    src_global = gd.collect(src)
    if gd.comm.rank == 0:
        ind = np.indices(src_global.shape)
        src_global += 1j * (ind[0] / 10. + ind[1] / 100. + ind[2] / 1000.)
        #src_global[1] += 0.5j
        print('GLOBAL ARRAY', src_global.shape)
        print(src_global.squeeze())
    gd.distribute(src_global, src)
    goal = gd2.zeros(dtype=float)
    goal[:] = gd.comm.rank # get_members()[gd2.comm.rank]
    goal_global = gd2.collect(goal)
    if gd.comm.rank == 0:
        print('GOAL GLOBAL')
        print(goal_global.squeeze())
    gd.comm.barrier()
    #return

    recvbuf = redistribute(gd, gd2, src, distribute_dir, reduce_dir,
                           operation='forth')
    recvbuf_master = gd2.collect(recvbuf)
    if gd2.comm.rank == 0:
        print('RECV')
        print(recvbuf_master)
        err = src_global - recvbuf_master
        print('MAXERR', np.abs(err).max())

    hopefully_orig = redistribute(gd, gd2, recvbuf, distribute_dir, reduce_dir,
                                  operation='back')
    tmp = gd.collect(hopefully_orig)
    if gd.comm.rank == 0:
        print('FINALLY')
        print(tmp)
        err2 = src_global - tmp
        print('MAXERR', np.abs(err2).max())


def test(N_c, gd, gd2, reduce_dir, distribute_dir, verbose=True):
    src = gd.zeros(dtype=complex)
    src[:] = gd.comm.rank

    #if gd.comm.rank == 0:
    #    print(gd)
        #print('hmmm', gd, gd2)

    src_global = gd.collect(src)
    if gd.comm.rank == 0:
        ind = np.indices(src_global.shape)
        src_global += 1j * (ind[0] / 10. + ind[1] / 100. + ind[2] / 1000.)
        #src_global[1] += 0.5j
        if verbose:
            print('GLOBAL ARRAY', src_global.shape)
            print(src_global)
    gd.distribute(src_global, src)
    goal = gd2.zeros(dtype=float)
    goal[:] = gd2.comm.rank
    goal_global = gd2.collect(goal)
    if gd.comm.rank == 0 and verbose:
        print('GOAL GLOBAL')
        print(goal_global)
    gd.comm.barrier()
    
    recvbuf = redistribute(gd, gd2, src, distribute_dir, reduce_dir,
                           operation='forth')
    recvbuf_master = gd2.collect(recvbuf)
    #if np.all(N_c == [10, 16, 24]):
    #    recvbuf_master[5,8,3] = 7
    maxerr = 0.0
    if gd2.comm.rank == 0:
        #if N_c[0] == 7:
        #    recvbuf_master[5, 0, 0] = 7
        #recvbuf_master[0,0,0] = 7
        err = src_global - recvbuf_master
        maxerr = np.abs(err).max()
        if verbose:
            print('RECV FORTH')
            print(recvbuf_master)
            print('MAXERR', maxerr)
    maxerr = gd.comm.sum(maxerr)
    assert maxerr == 0.0, 'bad values after distribute "forth"'

    recvbuf2 = redistribute(gd, gd2, recvbuf, distribute_dir, reduce_dir,
                            operation='back')

    final_err = gd.comm.sum(np.abs(src - recvbuf2).max())
    assert final_err == 0.0, 'bad values after distribute "back"'


def rigorous_testing():
    from itertools import product, permutations, cycle
    from gpaw.mpi import world
    #gridpointcounts = [1, 2, 3, 5, 7, 10, 16, 24, 37]
    gridpointcounts = [1, 2, 10, 16, 37]
    cpucounts = np.arange(1, world.size + 1)
    pbc = cycle(product([0, 1], [0, 1], [0, 1]))

    # This yields all possible parallelizations!
    for parsize_c in product(cpucounts, cpucounts, cpucounts):
        if np.prod(parsize_c) != world.size:
            continue

        # All possible grid point counts
        for N_c in product(gridpointcounts, gridpointcounts, gridpointcounts):

            # We simply can't be bothered to also do all possible
            # combinations with PBCs.  Trying every possible set of
            # boundary conditions at least ones should be quite fine
            # enough.
            pbc_c = next(pbc)
            for dirs in permutations([0, 1, 2]):
                independent_dir, distribute_dir, reduce_dir = dirs

                parsize2_c = list(parsize_c)
                parsize2_c[reduce_dir] = 1
                parsize2_c[distribute_dir] *= parsize_c[reduce_dir]
                parsize2_c = tuple(parsize2_c)
                assert np.prod(parsize2_c) == np.prod(parsize_c)

                try:
                    gd = GridDescriptor(N_c=N_c, pbc_c=pbc_c,
                                        cell_cv=0.2 * np.array(N_c),
                                        parsize=parsize_c)
                    gd2 = get_compatible_grid_descriptor(gd, distribute_dir,
                                                         reduce_dir)
                         
                    #gd2 = gd.new_descriptor(parsize=parsize2_c)
                except ValueError:  # Skip illegal distributions
                    continue

                if gd.comm.rank == 1:
                    #print(gd, gd2)
                    print('N_c=%s[%s] redist %s -> %s [ind=%d dist=%d red=%d]'
                          % (N_c, pbc_c, parsize_c, parsize2_c,
                             independent_dir, distribute_dir, reduce_dir))
                gd.comm.barrier()
                test(N_c, gd, gd2, reduce_dir, distribute_dir,
                     verbose=False)


if __name__ == '__main__':
    playground()
    #rigorous_testing()
