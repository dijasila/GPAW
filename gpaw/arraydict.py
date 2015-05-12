import numpy as np

class ArrayDict(dict):
    """Distributed dictionary of fixed-size, fixed-dtype arrays.

    Elements are initialized as empty numpy arrays.

    Unlike a normal dictionary, this class implements a strict loop ordering
    which is consistent with that of the underlying atom partition."""
    def __init__(self, partition, shapes_a, dtype=float, d=None):
        dict.__init__(self)
        self.partition = partition
        if callable(shapes_a):
            shapes_a = [shapes_a(a) for a in range(self.partition.natoms)]
        self.shapes_a = shapes_a # global
        self.dtype = dtype
        if d is None:
            for a in partition.my_indices:
                self[a] = np.empty(self.shapes_a[a], dtype=dtype)
        else:
            self.update(d)
        self.check_consistency()

    # copy() is dangerous since redistributions write back
    # into arrays, and redistribution of a copy could lead to bugs
    # if the copy suffers.  I think the redistribution code does not
    # cause such problems presently, but I have disabled to be safe for now.
    #  -askhl
    #
    #def copy(self):
    #    return ArrayDict(self.partition, self.shapes_a, self.dtype, self)

    def deepcopy(self):
        copy = ArrayDict(self.partition, self.shapes_a, self.dtype)
        for a in self:
            copy[a] = self[a].copy()
        return copy

    def update(self, d):
        dict.update(self, d)
        self.check_consistency()

    def __getitem__(self, a):
        value = dict.__getitem__(self, a)
        assert value.shape == self.shapes_a[a]
        assert value.dtype == self.dtype
        return value

    def __setitem__(self, a, value):
        assert value.shape == self.shapes_a[a], \
            'defined shape %s vs new %s' % (self.shapes_a[a], value.shape)
        assert value.dtype == self.dtype
        dict.__setitem__(self, a, value)

    def redistribute(self, partition):
        """Redistribute according to specified partition."""
        def get_empty(a):
            return np.empty(self.shapes_a[a])

        self.partition.redistribute(partition, self, get_empty)
        self.partition = partition  # Better with immutable partition?
        self.check_consistency()

    def check_consistency(self):
        k1 = set(self.partition.my_indices)
        k2 = set(dict.keys(self))
        assert k1 == k2, 'Required keys %s different from actual %s' % (k1, k2)
        for a, array in self.items():
            assert array.dtype == self.dtype
            assert array.shape == self.shapes_a[a], \
                'array shape %s vs specified shape %s' % (array.shape,
                                                          self.shapes_a[a])

    def flatten_to_array(self, axis=None):
        # We could also implement it as a contiguous buffer.
        if len(self) == 0:
            # XXXXXX how should we deal with globally or locally empty arrays?
            # This will probably lead to bugs unless we get all the
            # dimensions right.
            return np.empty(0, self.dtype)
        if axis is None:
            return np.concatenate([self[a].ravel()
                                   for a in self.partition.my_indices])
        else:
            # XXX self[a].shape must all be consistent except along axis
            return np.concatenate([self[a] for a in self.partition.my_indices],
                                  axis=axis)

    def unflatten_from_array(self, data):
        M1 = 0
        for a in self.partition.my_indices:
            M2 = M1 + np.prod(self.shapes_a[a])
            dst = self[a].ravel()
            dst[:] = data[M1:M2]

    # These functions enforce the same ordering as self.partition
    # when looping.
    def keys(self):
        return self.partition.my_indices

    def __iter__(self):
        for key in self.partition.my_indices:
            yield key

    def values(self):
        return [self[key] for key in self]

    def iteritems(self):
        for key in self:
            yield key, self[key]

    def items(self):
        return list(self.iteritems())

    #def broadcast(self, comm, root):
    #    serial_partition = self.as_serial()
    #    self.redistribute(serial_partition)
    #    buf = self.flatten_to_array()
    #    #comm.broadcast(buf, 

    #def to_parent_comm(self):
    #    new_partition = self.partition.to_parent_comm()
    #    arraydict = ArrayDict(new_partition, self.shapes_a, dtype=float)
    #    assert arraydict.partition.comm.size >= self.partition.comm.size
    #    
    #    print self, arraydict
    #    # XXXX ? unfinished...

    def __str__(self):
        tokens = []
        for key in sorted(self.keys()):
            shapestr = 'x'.join(map(str, self.shapes_a[key]))
            tokens.append('%s:%s' % (key, shapestr))
        text = ', '.join(tokens)
        return '%s@rank%d/%d {%s}' % (self.__class__.__name__,
                                      self.partition.comm.rank,
                                      self.partition.comm.size,
                                      text)
