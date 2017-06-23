from gpaw.matrix import Matrix


class ArrayWaveFunctions:
    def __init__(self, M, N, dtype, data, dist):
        self.matrix = Matrix(M, N, dtype, data, dist)
        self.comm_to_be_summed_over = None
        self.comm = None
        self.dtype = self.matrix.dtype

    def __len__(self):
        return len(self.matrix)

    def finish_sumssss(self):
        if self.comm_to_be_summed_over and not self.transposed:
            self.comm_to_be_summed_over.sum(self.a, 0)
        self.comm_to_be_summed_over = None

    def multiply(self, alpha, opa, b, opb, beta, c):
        self.matrix.multiply(alpha, opa, b.matrix, opb, beta, c)
        if opa in 'NC' and self.comm:
            c.comm_to_be_summed_over = self.comm
            assert opb in 'TH' and b.comm is self.comm

    def matrix_elements(self, other, out=None, hermitian=False):
        if out is None:
            out = Matrix(len(self), len(other))
        self.multiply(self.dv, 'C', other, 'T', 0.0, out)
        return out

    def apply(self, func, out):
        func(self.array, out.array)

    def __setitem__(self, i, x):
        assert i == slice(None)
        x.eval(self.matrix)

    def __iadd__(self, other):
        other.eval(self.matrix, 1.0)
        return self

    def eval(self, matrix):
        matrix.array[:] = self.matrix.array


class UniformGridWaveFunctions(ArrayWaveFunctions):
    def __init__(self, nbands, gd, dtype=None, data=None, kpt=None, dist=None,
                 spin=0, collinear=True):
        ngpts = gd.get_size_of_global_array().prod()
        ArrayWaveFunctions.__init__(self, nbands, ngpts, dtype, data, dist)

        M = self.matrix

        if data is None:
            M.array = M.array.reshape(-1).reshape(M.dist.shape)
            M.transposed = False

        self.myshape = (M.dist.shape[0],) + tuple(gd.n_c)
        self.gd = gd
        self.dv = gd.dv
        self.comm = gd.comm

    @property
    def array(self):
        return self.matrix.array.reshape(self.myshape)

    def __repr__(self):
        s = ArrayWaveFunctions.__repr__(self).split('(')[1][:-1]
        shape = self.gd.get_size_of_global_array()
        s = 'UniformGridWaveFunctions({}, gpts={}x{}x{})'.format(s, *shape)
        return s

    def new(self, buf=None):
        return UniformGridWaveFunctions(len(self), self.gd, self.dtype,
                                        buf,
                                        None, self.matrix.dist)

    def plot(self):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111)
        a, b, c = self.array.shape[1:]
        ax.plot(self.array[0, a // 2, b // 2])
        plt.show()


class PlaneWaveExpansionWaveFunctions(ArrayWaveFunctions):
    pass


"""
    def __init__(self, M, pd, data, q=-1, dist=None):
        orig = data
        if pd.dtype == float:
            data = data.view(float)
        ArrayWaveFunctions.__init__(self, M, data.shape[1], pd.dtype,
                                    data, dist)
        self.array = orig
        self.pd = pd
        self.q = q
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def __getitem______________________(self, i):
        assert self.distribution.shape[0] == self.shape[0]
        return self.array[i]

    def new(self, buf=None):
        if buf is not None:
            buf = buf.ravel()[:self.array.size]
            buf.shape = self.array.shape
        return PWExpansionMatrix(self.shape[0], self.pd, buf, self.q,
                                 self.dist)

    def multiply(self, alpha, opa, b, opb, beta, c):
        if (self.pd.dtype == float and opa in 'NC' and
            isinstance(b, PWExpansionMatrix)):
            assert opa == 'C' and opb == 'T' and beta == 0
            a = self
            c.a[:] = np.dot(a.a, b.a.T)
            c.a *= 2 * alpha
            c.a -= alpha * np.outer(a.a[:, 0], b.a[:, 0])
        else:
            SpatialMatrix.multiply(self, alpha, opa, b, opb, beta, c)
"""
