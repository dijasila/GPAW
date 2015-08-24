import numpy as np

def accordion_redistribute(gd, src, axis, operation='forth'):
    """Redistribute grid longitudinally to uniform blocksize.

    Redistributes a grid along one axis from GPAW's standard
    non-uniform block distribution to one with fixed blocksize except
    for last element which may be smaller.

    For example along one axis, GPAW may have the following blocks:

      [5, 5, 6, 5, 6, 5].

    This would be redistributed to
    
      [6, 6, 6, 6, 6, 2].
    
    This makes the array compatible with some parallel Fourier
    transform libraries such as FFTW-MPI or PFFT.

    This probably involves relatively little communication and so we do
    not care to fiercely optimize this."""
    parsize = gd.parsize_c[axis]
    ngpts = gd.N_c[axis]  # XXX pbc?
    assert all(gd.pbc_c)
    blocksize = -(-ngpts // parsize)
    remainder = ngpts - (parsize - 1) * blocksize
    n_p = gd.n_cp[axis]
    
    forward = (operation == 'forth')
    assert forward or operation == 'back'
    assert (parsize - 1) * blocksize + remainder == ngpts
    assert remainder <= blocksize
    assert len(gd.n_cp[axis]) == parsize + 1
    ngpts_p = np.empty(parsize + 1, dtype=int)
    ngpts_p[0] = 0 if gd.pbc_c[axis] else 1
    ngpts_p[1:-1] = blocksize
    ngpts_p[-1] = remainder
    n2_p = np.cumsum(ngpts_p)
    n2_cp = list(gd.n_cp)
    n2_cp[axis] = n2_p

    gd2 = gd.new_descriptor(n_cp=n2_cp)
    peer_comm = gd.get_axial_communicator(axis)

    shape = gd.n_c.copy()
    if peer_comm.rank == peer_comm.size - 1:
        shape[axis] = remainder
    else:
        shape[axis] = blocksize
    if operation == 'forth':
        dst = np.empty(shape, dtype=src.dtype)
    else:
        dst = gd.empty(dtype=src.dtype)
    dst.fill(-2)

    from gpaw.utilities.partition import AtomPartition
    from gpaw.arraydict import ArrayDict

    rank1_a = np.empty(ngpts, dtype=int)  # pbc?
    rank2_a = np.empty(ngpts, dtype=int)  # pbc?
    rank1_a.fill(-1)
    rank2_a.fill(-1)
    for i in range(peer_comm.size):
        rank1_a[n_p[i]:n_p[i + 1]] = i
        rank2_a[blocksize * i:blocksize * (i + 1)] = i
        # (Note that on last rank, above slice may be shorter than blocksize)
    assert (rank1_a >= 0).all(), str(rank1_a)
    assert (rank2_a >= 0).all(), str(rank2_a)
    partition1 = AtomPartition(peer_comm, rank1_a)
    partition2 = AtomPartition(peer_comm, rank2_a)
    
    # The plan is to divide things in chunks of 1xNxM for simplicity.
    # Probably not very efficient but this is meant for layouts that
    # are "almost correct" in the first place, requiring only a few
    # adjustments.
    shape = gd.n_c.copy()
    shape[axis] = 1
    shapes_a = [tuple(shape)] * ngpts

    if forward:
        data = ArrayDict(partition1, shapes_a, dtype=src.dtype)
    else:
        data = ArrayDict(partition2, shapes_a, dtype=src.dtype)

    slices = [slice(None, None, None)] * 3

    def grid_to_dict(arr):
        for i, g in enumerate(range(gd.beg_c[axis], gd.end_c[axis])):
            slices[axis] = i
            dictslice = data[g]
            arrayslice = arr[slices[0], slices[1], slices[2]]
            dictslice[:] = arrayslice

    def dict_to_grid(arr):
        globalstart = peer_comm.rank * blocksize
        if peer_comm.rank == peer_comm.size - 1:
            globalstop = globalstart + remainder
        else:
            globalstop = globalstart + blocksize
        for i, g in enumerate(range(globalstart, globalstop)):
            slices[axis] = i
            dictslice = data[g]
            arrayslice = arr[slices[0], slices[1], slices[2]]
            arrayslice[:] = dictslice

    grid_to_dict(src)
    data.redistribute(partition2 if forward else partition1)
    dict_to_grid(dst)

    return gd2, dst
            

def playground():
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import world
    N_c = np.array((12, 1, 4))
    gd = GridDescriptor(N_c, cell_cv=0.2 * N_c,
                        parsize=(world.size, 1, 1))
    print gd
    src = gd.zeros()
    src[:] = gd.comm.rank
    
    gd2, dst = accordion_redistribute(gd, src, axis=0, operation='forth')

    gd_, orig = accordion_redistribute(gd, dst, axis=0, operation='back')

    grumble = gd2.collect(dst)

    src0 = gd.collect(src)

    if gd.comm.rank == 0:
        print src0.squeeze()

    if gd2.comm.rank == 0:
        print grumble.squeeze()

    #if gd.comm.rank == 1:
    #    print dst

if __name__ == '__main__':
    playground()
