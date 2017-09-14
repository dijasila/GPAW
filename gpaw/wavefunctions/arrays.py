import numpy as np

from gpaw.matrix import Matrix


class MatrixInFile:
    def __init__(self, M, N, dtype, data, dist):
        self.shape = (M, N)
        self.dtype = dtype
        self.data = data
        self.dist = dist

    def read(self):
        matrix = Matrix(*self.shape, self.dtype, dist=self.dist)
        # Read band by band to save memory
        for myn, psit_G in enumerate(matrix.data):
            n = self.bd.global_index(myn)
            if self.gd.comm.rank == 0:
                big_psit_G = np.asarray(self.data[n], self.dtype)
            else:
                big_psit_G = None
            self.gd.distribute(big_psit_G, psit_G)
        return matrix


class ArrayWaveFunctions:
    def __init__(self, M, N, dtype, data, dist):
        if data is None or isinstance(data, np.ndarray):
            self.matrix = Matrix(M, N, dtype, data, dist)
            self.in_memory = True
        else:
            self.matrix = MatrixInFile(M, N, dtype, data, dist)
            self.in_memory = False
        self.comm_to_be_summed_over = None
        self.comm = None
        self.dtype = self.matrix.dtype

    def __len__(self):
        return len(self.matrix)

    def read_from_file(self):
        self.matrix = self.matrix.read()
        self.in_memory = True

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
            out = Matrix(len(self), len(other), dtype=self.dtype)
        self.multiply(self.dv, 'C', other, 'T', 0.0, out)
        return out

    def apply(self, func, out):
        func(self.array, out.array)

    def __setitem__(self, i, x):
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
        self.kpt = kpt
        self.spin = spin
        self.comm = gd.comm

    @property
    def array(self):
        return self.matrix.array.reshape(self.myshape)

    def __repr__(self):
        s = ArrayWaveFunctions.__repr__(self).split('(')[1][:-1]
        shape = self.gd.get_size_of_global_array()
        s = 'UniformGridWaveFunctions({}, gpts={}x{}x{})'.format(s, *shape)
        return s

    def new(self, buf=None, dist='inherit', nbands=None):
        if dist == 'inherit':
            dist = self.matrix.dist
        return UniformGridWaveFunctions(nbands or len(self),
                                        self.gd, self.dtype,
                                        buf,
                                        self.kpt, dist,
                                        self.spin)

    def view(self, n1, n2):
        return UniformGridWaveFunctions(n2 - n1, self.gd, self.dtype,
                                        self.array[n1:n2],
                                        self.kpt, None,
                                        self.spin)

    def plot(self):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111)
        a, b, c = self.array.shape[1:]
        ax.plot(self.array[0, a // 2, b // 2])
        plt.show()


class PlaneWaveExpansionWaveFunctions(ArrayWaveFunctions):
    def __init__(self, nbands, pd, dtype=None, data=None, kpt=None, dist=None,
                 spin=0, collinear=True):
        if data is None:
            data = pd.empty(nbands, q=kpt)
        self.array = data
        if pd.dtype == float:
            data = data.view(float)
        mynbands, ng = data.shape
        ArrayWaveFunctions.__init__(self, nbands, ng, dtype, data, dist)
        self.pd = pd
        self.dv = pd.gd.dv / pd.gd.N_c.prod()
        self.kpt = kpt
        self.spin = spin
        self.myshape = (self.matrix.dist.shape[0], ng)

    def multiply(self, alpha, opa, b, opb, beta, c):
        if opa == 'C' and opb == 'T' and beta == 0:
            if self.pd.dtype == complex:
                ArrayWaveFunctions.multiply(self, alpha, opa, b, opb, beta, c)
            else:
                ArrayWaveFunctions.multiply(self, 2 * alpha,
                                            opa, b, opb, beta, c)
                c.array -= alpha * np.outer(self.matrix.array[:, 0],
                                            b.matrix.array[:, 0])
        else:
            1 / 0

    def new(self, buf=None, dist='inherit', nbands=None):
        if buf is not None:
            array = self.array
            buf = buf.ravel()[:array.size]
            buf.shape = array.shape
        if dist == 'inherit':
            dist = self.matrix.dist
        return PlaneWaveExpansionWaveFunctions(nbands or len(self),
                                               self.pd, self.dtype,
                                               buf,
                                               self.kpt, dist,
                                               self.spin)

    def view(self, n1, n2):
        return PlaneWaveExpansionWaveFunctions(n2 - n1, self.pd, self.dtype,
                                               self.array[n1:n2],
                                               self.kpt, None,
                                               self.spin)
