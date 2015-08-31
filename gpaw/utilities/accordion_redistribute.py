from __future__ import print_function
import numpy as np


def accordion_redistribute(gd, src, axis, operation='forth'):
    """Redistribute grid longitudinally to uniform blocksize.

    Redistributes a grid along one axis from GPAW's standard
    non-uniform block distribution to one with fixed blocksize except
    for last element which may be smaller.

    For example along one axis, GPAW may distribute in blocks as follows:

      [5, 5, 6, 5, 6, 5].

    This would be redistributed to

      [6, 6, 6, 6, 6, 2].

    This makes the array compatible with some parallel Fourier
    transform libraries such as FFTW-MPI or PFFT.

    This probably involves relatively little communication and so we do
    not care to fiercely optimize this."""
    if axis != 0:
        raise NotImplementedError('accordion redistribute along non-x axis')

    parsize = gd.parsize_c[axis]
    ngpts = gd.get_size_of_global_array()[axis]
    blocksize = -(-ngpts // parsize)
    remainder = ngpts - (parsize - 1) * blocksize
    if remainder < 0:
        raise BadGridError('Dimensions incompatible (grid too small)')
    n_p = gd.n_cp[axis] - gd.n_cp[axis][0]

    forward = (operation == 'forth')
    assert forward or operation == 'back'
    assert (parsize - 1) * blocksize + remainder == ngpts
    assert remainder <= blocksize
    assert len(gd.n_cp[axis]) == parsize + 1
    ngpts_p = np.empty(parsize + 1, dtype=int)
    ngpts_p[0] = 0
    ngpts_p[1:-1] = blocksize
    ngpts_p[-1] = remainder

    peer_comm = gd.get_axial_communicator(axis)

    if peer_comm.rank == peer_comm.size - 1:
        myblocksize = remainder
    else:
        myblocksize = blocksize

    shape = gd.n_c.copy()
    shape[axis] = myblocksize
    if forward:
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

    grrr = gd.n_cp[axis][0]

    def grid_to_dict(arr):
        if forward:
            beg = gd.beg_c[axis] - grrr
            end = gd.end_c[axis] - grrr
        else:
            beg = blocksize * peer_comm.rank
            end = beg + myblocksize

        for i, g in enumerate(range(beg, end)):
            slices[axis] = i
            dictslice = data[g]
            arrayslice = arr[slices[0], slices[1], slices[2]]
            dictslice[:] = arrayslice

    def dict_to_grid(arr):
        if forward:
            globalstart = peer_comm.rank * blocksize
            globalstop = globalstart + myblocksize
        else:
            globalstart = gd.beg_c[axis] - grrr
            globalstop = globalstart + gd.n_c[axis]
        for i, g in enumerate(range(globalstart, globalstop)):
            slices[axis] = i
            dictslice = data[g]
            arrayslice = arr[slices[0], slices[1], slices[2]]
            arrayslice[:] = dictslice

    grid_to_dict(src)
    data.redistribute(partition2 if forward else partition1)
    dict_to_grid(dst)

    return dst
            

def playground():
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import world
    N_c = np.array((12, 1, 4))
    gd = GridDescriptor(N_c, cell_cv=0.2 * N_c,
                        parsize=(world.size, 1, 1))
    print(gd)
    src = gd.zeros()
    src[:] = gd.comm.rank
    dst = accordion_redistribute(gd, src, axis=0, operation='forth')
    orig = accordion_redistribute(gd, dst, axis=0, operation='back')
    src0 = gd.collect(src)


class BadGridError(ValueError):
    pass


def test(N_c, pbc_c, parsize_c, axis):
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import world
    N_c = np.array(N_c)
    try:
        gd = GridDescriptor(N_c, cell_cv=0.2 * N_c, pbc_c=pbc_c,
                            parsize=parsize_c)
    except ValueError as e:
        raise BadGridError(e)
    src = gd.zeros()
    src[:] = gd.comm.rank
    dst = accordion_redistribute(gd, src, axis=axis, operation='forth')
    orig = accordion_redistribute(gd, dst, axis=axis, operation='back')
    err = np.abs(src - orig).max()
    
    if err == 0.0:
        status = 'OK:  '
    else:
        status = 'Bad: '
    msg = '%s N_c=%s pbc_c=%s parsize_c=%s' % (status, N_c, pbc_c,
                                               gd.parsize_c)
    #assert err == 0.0, 'Bad: N_c=%s pbc_c=%s' % (N_c, pbc_c)
    if world.rank == 0:
        print(msg)
    
    # Let's just raise an error properly...
    assert err == 0.0, msg


def test_thoroughly():
    from itertools import product
    from gpaw.mpi import world
    Nvalues = [1, 2, 5, 17, 37]
    pbcvalues = [False, True]
    
    cpucounts = np.arange(1, world.size + 1)

    for parsize_c in product(cpucounts, cpucounts, cpucounts):
        if np.prod(parsize_c) != world.size:
            continue
    
        for N_c in product(Nvalues, Nvalues, Nvalues):
            for pbc_c in product(pbcvalues, pbcvalues, pbcvalues):
                try:
                    test(N_c, pbc_c, parsize_c, 0)
                except BadGridError:
                    pass


if __name__ == '__main__':
    test_thoroughly()
    #playground()
