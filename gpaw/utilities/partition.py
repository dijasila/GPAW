import numpy as np


class EvenPartitioning:
    """Represents an even partitioning of N elements over a communicator.

    For example N=17 and comm.size=5 will result in this distribution:

     * rank 0 has 3 local elements: 0, 1, 2
     * rank 1 has 3 local elements: 3, 4, 5
     * rank 2 has 3 local elements: 6, 7, 8
     * rank 3 has 4 local elements: 9, 10, 11, 12
     * rank 4 has 4 local elements: 13, 14, 15, 16

    This class uses only the 'rank' and 'size' communicator attributes."""
    def __init__(self, comm, N):
        # Conventions:
        #  n, N: local/global size
        #  i, I: local/global index
        self.comm = comm
        self.N = N
        self.nlong = -(-N // comm.size) # size of a 'long' slice
        self.nshort = N // comm.size # size of a 'short' slice
        self.longcount = N % comm.size # number of ranks with a 'long' slice
        self.shortcount = comm.size - self.longcount # ranks with 'short' slice

    def nlocal(self, rank=None):
        """Get the number of locally stored elements."""
        if rank is None:
            rank = self.comm.rank
        if rank < self.shortcount:
            return self.nshort
        else:
            return self.nlong

    def minmax(self, rank=None):
        """Get the minimum and maximum index of elements stored locally."""
        if rank is None:
            rank = self.comm.rank
        I1 = self.nshort * rank
        if rank < self.shortcount:
            I2 = I1 + self.nshort
        else:
            I1 += rank - self.shortcount
            I2 = I1 + self.nlong
        return I1, I2

    def slice(self, rank=None):
        """Get the list of indices of locally stored elements."""
        I1, I2 = self.minmax(rank=rank)
        return np.arange(I1, I2)

    def global2local(self, I):
        """Get a tuple (rank, local index) from global index I."""
        nIshort = self.nshort * self.shortcount
        if I < nIshort:
            return I // self.nshort, I % self.nshort
        else:
            Ioffset = I - nIshort
            return self.shortcount + Ioffset // self.nlong, Ioffset % self.nlong

    def local2global(self, i, rank=None):
        """Get global index I corresponding to local index i on rank."""
        if rank is None:
            rank = self.comm.rank
        return rank * self.nshort + max(rank - self.shortcount, 0) + i

    def as_atom_partition(self):
        rank_a = [self.global2local(i)[0] for i in range(self.N)]
        return AtomPartition(self.comm, rank_a)

    def get_description(self):
        lines = []
        for a in range(self.comm.size):
            elements = ', '.join(map(str, self.slice(a)))
            line = 'rank %d has %d local elements: %s' % (a, self.nlocal(a),
                                                          elements)
            lines.append(line)
        return '\n'.join(lines)

# Interface for things that can be redistributed with general_redistribute
class Redistributable:
    def new_buffer(self, a): raise NotImplementedError
    def get_array(self, a): raise NotImplementedError
    def assign(self, a): raise NotImplementedError

# Let's keep this as an independent function for now in case we change the
# classes 5 times, like we do
def general_redistribute(comm, src_rank_a, dst_rank_a, redistributable):
    # To do: it should be possible to specify duplication to several ranks
    # But how is this done best?
    requests = []
    flags = (src_rank_a != dst_rank_a)
    my_incoming_atom_indices = np.argwhere(np.bitwise_and(flags, \
        dst_rank_a == comm.rank)).ravel()
    my_outgoing_atom_indices = np.argwhere(np.bitwise_and(flags, \
        src_rank_a == comm.rank)).ravel()

    for a in my_incoming_atom_indices:
        # Get matrix from old domain:
        buf = redistributable.new_buffer(a)
        requests.append(comm.receive(buf, src_rank_a[a], tag=a, block=False))
        # These arrays are not supposed to pointers into a larger,
        # contiguous buffer, so we should make a copy - except we
        # must wait until we have completed the send/receiving
        # into them, so we will do it a few lines down.
        redistributable.assign(a, buf)

    for a in my_outgoing_atom_indices:
        # Send matrix to new domain:
        buf = redistributable.get_array(a)
        requests.append(comm.send(buf, dst_rank_a[a], tag=a, block=False))
                                  
    comm.waitall(requests)


class AtomPartition:
    """Represents atoms distributed on a standard grid descriptor."""
    def __init__(self, comm, rank_a):
        self.comm = comm
        self.rank_a = np.array(rank_a)
        self.my_indices = self.get_indices(comm.rank)
    
    def get_indices(self, rank):
        return np.where(self.rank_a == rank)[0]

    def to_parent_comm(self):
        parent_rank_a = self.comm.get_members()[self.rank_a]
        return AtomPartition(self.comm.parent, parent_rank_a)

    def to_even_distribution(self, atomdict_ax, get_empty):
        even_part = EvenPartitioning(self.comm,
                                     len(self.rank_a)).as_atom_partition()
        self.redistribute(even_part, atomdict_ax, get_empty)

    def from_even_distribution(self, atomdict_ax, get_empty):
        even_part = EvenPartitioning(self.comm,
                                     len(self.rank_a)).as_atom_partition()
        even_part.redistribute(self, atomdict_ax, get_empty)

    def redistribute(self, new_partition, atomdict_ax, get_empty):
        assert self.comm == new_partition.comm
        # atomdict_ax may be a dictionary or a list of dictionaries
        has_many = not hasattr(atomdict_ax, 'items')
        if has_many:
            class Redist:
                def new_buffer(self, a):
                    return get_empty(a)
                def assign(self, a, b_x):
                    for u, d_ax in enumerate(atomdict_ax):
                        assert a not in d_ax
                        atomdict_ax[u][a] = b_x[u]
                def get_array(self, a):
                    return np.array([d_ax.pop(a) for d_ax in atomdict_ax])
        else:
            class Redist:
                def new_buffer(self, a):
                    return get_empty(a)
                def assign(self, a, b_x):
                    assert a not in atomdict_ax
                    atomdict_ax[a] = b_x
                def get_array(self, a):
                    return atomdict_ax.pop(a)

        general_redistribute(self.comm, self.rank_a,
                             new_partition.rank_a, Redist())
