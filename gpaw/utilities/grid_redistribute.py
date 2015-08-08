import numpy as np
from gpaw.grid_descriptor import GridDescriptor


def redistribute(gd, gd2, src, distribute_dir, reduce_dir, operation='forth',
                 nasty=False):
    """Perform certain simple redistributions among two grid descriptors.

    Redistribute src from gd with decomposition X x Y x Z to gd2 with
    decomposition X x YZ x 1, or some variation of this.  We say that
    we "reduce" along Z while we "distribute" along Y.  The
    redistribution is one-to-one.

    gd and gd2 must have the same parallelization in the third direction.

    reduce_dir is the direction (0, 1, or 2) in which gd2 is serial,
    and distribute_dir is the direction in which gd2 has more cores
    than gd.

    Presently the only implemented cases are:
        distribute_dir = 1 and reduce_dir = 2
        distribute_dir = 0 and reduce_dir = 1
    """
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
    if not forward:
        raise NotImplementedError('Sorry, no way back yet.')

    dirs = (independent_dir, distribute_dir, reduce_dir)
    if not nasty and dirs in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 1, 0)]:
        raise NotImplementedError('Cannot reduce dir %d and distribute dir %d'
                                  % (reduce_dir, distribute_dir))
    # OK dirs: (0, 1, 2) and (1, 2, 0).
    #
    # In principle it is possible to fix (0, 2, 1) and (2, 1, 0)
    # because they are similar operations along trivially different
    # axes.

    dtype = src.dtype

    # Construct a communicator consisting of all those processes that
    # participate in domain decomposition along the reduction
    # direction.
    #
    # All necessary communication can be done within that
    # subcommunicator using MPI alltoallv.
    pos_c = gd.parpos_c.copy()
    peer_ranks = []
    for i in range(gd.parsize_c[reduce_dir]):
        pos_c[reduce_dir] = i
        peer_ranks.append(gd.get_rank_from_processor_position(pos_c))
    peer_comm = gd.comm.new_communicator(peer_ranks)
    members = peer_comm.get_members()

    sendnpts_rdir = gd.n_c[reduce_dir]
    recvnpts_ddir = gd2.n_c[distribute_dir]
    npts_idir = gd.n_c[independent_dir]
    assert npts_idir == gd2.n_c[independent_dir]

    sendn_p = gd2.n_cp[distribute_dir]
    recvn_p = gd.n_cp[reduce_dir]

    recvbuf = gd2.zeros(dtype=dtype).ravel()
    recvbuf[:] = -3

    # We want this to work no matter which direction is distribute and
    # reduce.  But that is tricky to code.  So we use a standard order
    # of the three.
    #
    # Thus we have to always transpose the src/dst arrays consistently
    # when interacting with the contiguous MPI send/recv buffers.
    dirs = (independent_dir, distribute_dir, reduce_dir)

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
    
    dst = gd2.zeros(dtype=dtype)
    sendchunks = []
    recvchunks = []
    recv_chunk_copiers = []
    
    class ChunkCopier:
        def __init__(self, i, start, stop):
            self.chunk = recvchunks[i]
            self.dstchunk = dst.transpose(*dirs)[:, :, start:stop]

        def copy(self):
            self.dstchunk.flat[:] = self.chunk

    recvchunk_start = 0
    for i in range(peer_comm.size):
        parent_rank = members[i]
        parent_src_coord = \
            gd.get_processor_position_from_rank(parent_rank)[reduce_dir]
        parent_dst_coord = \
            gd2.get_processor_position_from_rank(parent_rank)[distribute_dir]

        sendstart_ddir = sendn_p[parent_dst_coord] - gd.beg_c[distribute_dir]
        sendstop_ddir = sendn_p[parent_dst_coord + 1] \
            - gd.beg_c[distribute_dir]
        sendnpts_ddir = sendstop_ddir - sendstart_ddir
        
        recvstart_rdir = recvn_p[parent_src_coord]
        recvstop_rdir = recvn_p[parent_src_coord + 1]
        recvnpts_rdir = recvstop_rdir - recvstart_rdir

        # Grab subarray that is going to be sent to process i.
        sendchunk = src.transpose(*dirs)[:, sendstart_ddir:sendstop_ddir, :]

        assert sendchunk.size == sendnpts_rdir * sendnpts_ddir * npts_idir
        sendchunks.append(sendchunk)

        recvchunksize = recvnpts_rdir * recvnpts_ddir * npts_idir
        recvchunk = recvbuf[recvchunk_start:recvchunk_start
                            + recvchunksize]
        recvchunks.append(recvchunk)
        recvchunk_start += recvchunksize
        
        recv_chunk_copiers.append(ChunkCopier(i, recvstart_rdir,
                                              recvstop_rdir))

    sendcounts = np.array([chunk.size for chunk in sendchunks])
    recvcounts = np.array([chunk.size for chunk in recvchunks])
    # Parallel Ole Holm-Nielsen check:
    # (First call int because some versions of numpy return np.intXX
    #  which does not trigger single-number comm.sum)
    nsend = int(sendcounts.sum())
    nrecv = int(recvcounts.sum())
    assert peer_comm.sum(nsend) == peer_comm.sum(nrecv)
    senddispls = np.array([0] + list(np.cumsum(sendcounts))[:-1], dtype=int)
    recvdispls = np.array([0] + list(np.cumsum(recvcounts))[:-1], dtype=int)

    sendbuf = np.concatenate([sendchunk.ravel() for sendchunk in sendchunks])

    peer_comm.alltoallv(sendbuf, sendcounts, senddispls,
                        recvbuf, recvcounts, recvdispls)

    # Copy contiguous blocks of receive buffer back into precoded slices:
    for chunk_copier in recv_chunk_copiers:
        chunk_copier.copy()
    return dst


def playground():
    np.set_printoptions(linewidth=176)
    N_c = [1, 5, 7]

    pbc_c = (1, 1, 1)

    distribute_dir = 1
    reduce_dir = 2

    parsize_c = (1, 2, 2)

    parsize2_c = list(parsize_c)
    parsize2_c[reduce_dir] = 1
    parsize2_c[distribute_dir] *= parsize_c[reduce_dir]
    assert np.prod(parsize2_c) == np.prod(parsize_c)

    gd = GridDescriptor(N_c=N_c, pbc_c=pbc_c, cell_cv=0.2 * np.array(N_c),
                        parsize=parsize_c)
    gd2 = gd.new_descriptor(parsize=parsize2_c)

    src = gd.zeros(dtype=complex)
    src[:] = gd.comm.rank

    src_global = gd.collect(src)
    if gd.comm.rank == 0:
        ind = np.indices(src_global.shape)
        src_global += 1j * (ind[0] / 10. + ind[1] / 100. + ind[2] / 1000.)
        #src_global[1] += 0.5j
        print 'GLOBAL ARRAY', src_global.shape
        print src_global
    gd.distribute(src_global, src)
    goal = gd2.zeros(dtype=float)
    goal[:] = gd2.comm.rank
    goal_global = gd2.collect(goal)
    if gd.comm.rank == 0:
        print 'GOAL GLOBAL'
        print goal_global
    gd.comm.barrier()

    recvbuf = redistribute(gd, gd2, src, distribute_dir, reduce_dir,
                           nasty=True)
    recvbuf_master = gd2.collect(recvbuf)
    if gd2.comm.rank == 0:
        print 'RECV'
        print recvbuf_master
        err = src_global - recvbuf_master
        print 'MAXERR', np.abs(err).max()


def test(N_c, gd, gd2, reduce_dir, distribute_dir, verbose=True):
    src = gd.zeros(dtype=complex)
    src[:] = gd.comm.rank

    src_global = gd.collect(src)
    if gd.comm.rank == 0:
        ind = np.indices(src_global.shape)
        src_global += 1j * (ind[0] / 10. + ind[1] / 100. + ind[2] / 1000.)
        #src_global[1] += 0.5j
        if verbose:
            print 'GLOBAL ARRAY', src_global.shape
            print src_global
    gd.distribute(src_global, src)
    goal = gd2.zeros(dtype=float)
    goal[:] = gd2.comm.rank
    goal_global = gd2.collect(goal)
    if gd.comm.rank == 0 and verbose:
        print 'GOAL GLOBAL'
        print goal_global
    gd.comm.barrier()
    
    recvbuf = redistribute(gd, gd2, src, distribute_dir, reduce_dir)
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
            print 'RECV'
            print recvbuf_master
            print 'MAXERR', maxerr
    maxerr = gd.comm.sum(maxerr)
    return maxerr


def rigorous_testing():
    from itertools import product, permutations
    from gpaw.mpi import world
    #gridpointcounts = [1, 2, 3, 5, 7, 10, 16, 24, 37]
    gridpointcounts = [1, 2, 5, 10, 16, 37]
    cpucounts = np.arange(1, world.size + 1)

    failures = []

    # This yields all possible parallelizations!
    for parsize_c in product(cpucounts, cpucounts, cpucounts):
        if np.prod(parsize_c) != world.size:
            continue

        # All possible grid point counts
        for N_c in product(gridpointcounts, gridpointcounts, gridpointcounts):
            pbc_c = (1, 1, 1)

            for dirs in permutations([0, 1, 2]):
                independent_dir, distribute_dir, reduce_dir = dirs

                # Skip known errors
                if 1:
                    if dirs == (0, 2, 1):
                        continue
                    if dirs == (1, 0, 2):
                        continue
                    if dirs == (1, 2, 0):
                        continue
                    if dirs == (2, 1, 0):
                        continue

                parsize2_c = list(parsize_c)
                parsize2_c[reduce_dir] = 1
                parsize2_c[distribute_dir] *= parsize_c[reduce_dir]
                parsize2_c = tuple(parsize2_c)
                assert np.prod(parsize2_c) == np.prod(parsize_c)

                try:
                    gd = GridDescriptor(N_c=N_c, pbc_c=pbc_c,
                                        cell_cv=0.2 * np.array(N_c),
                                        parsize=parsize_c)
                    gd2 = gd.new_descriptor(parsize=parsize2_c)
                except ValueError:  # Skip illegal distributions
                    continue

                try:
                    maxerr = test(N_c, gd, gd2, reduce_dir, distribute_dir,
                                  verbose=False)
                    if maxerr == 0.0:
                        result = 'OK'
                    else:
                        result = 'FAIL'
                except AssertionError:
                    result = 'FAIL'

                if result == 'FAIL':
                    failures.append((parsize_c, N_c, dirs))
                if gd.comm.rank == 0:
                    print('N_c=%s redist %s -> %s [ind=%d red=%d dist=%d]: %s'
                          % (N_c, parsize_c, parsize2_c, independent_dir,
                             reduce_dir, distribute_dir, result))

    if gd.comm.rank == 0:
        print
        print 'Failures'
        print '--------'
        for parsize, N_c, dirs in failures:
            print 'parsize=%s N=%s (ind dist red)=%s' % (parsize, N_c, dirs)

    return failures


if __name__ == '__main__':
    playground()
    #rigorous_testing()
