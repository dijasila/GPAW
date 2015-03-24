import numpy as np

class ArrayDict(dict):
    def __init__(self, partition, shapes_a, dtype=float, d=None):
        dict.__init__(self)
        self.partition = partition
        self.shapes_a = shapes_a # global
        self.dtype = dtype
        if d is None:
            for a in partition.my_indices:
                self[a] = np.empty(self.shapes_a[a], dtype=dtype)
        else:
            self.update(d)
        self.check_consistency()

    def copy(self):
        return ArrayDict(self.partition, self.shapes_a, self.dtype, self)

    def deepcopy(self):
        copy = Arraydict(self.partition, self.shapes_a, self.dtype)
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
        def get_empty(a):
            return np.empty(self.shapes_a[a])

        self.partition.redistribute(partition, self, get_empty)
        self.partition = partition  # Better with immutable partition?
        self.check_consistency()

    def check_consistency(self):
        k1 = set(self.keys())
        k2 = set(self.partition.my_indices)
        assert k1 == k2, 'Inconsistent keys %s vs from %s' % (k1, k2)
        for a, array in self.items():
            assert array.dtype == self.dtype
            assert array.shape == self.shapes_a[a], \
                'array shape %s vs specified shape %s' % (array.shape,
                                                          self.shapes_a[a])

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
