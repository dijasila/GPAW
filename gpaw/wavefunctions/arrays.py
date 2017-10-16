import numpy as np

from gpaw.matrix import Matrix, create_distribution


class MatrixInFile:
    def __init__(self, M, N, dtype, data, dist):
        self.shape = (M, N)
        self.dtype = dtype
        self.array = data
        self.dist = create_distribution(M, N, *dist)

    def read(self, gd, shape):
        matrix = Matrix(*self.shape, dtype=self.dtype, dist=self.dist)
        # Read band by band to save memory
        for myn, psit_G in enumerate(matrix.array):
            n = self.dist.global_index(myn)
            if gd.comm.rank == 0:
                big_psit_G = np.asarray(self.array[n], self.dtype)
            else:
                big_psit_G = None
            gd.distribute(big_psit_G, psit_G.reshape(shape))

        return matrix


class ArrayWaveFunctions:
    def __init__(self, M, N, dtype, data, dist):
        if data is None or isinstance(data, np.ndarray):
            self.matrix = Matrix(M, N, dtype, data, dist)
            self.in_memory = True
        else:
            self.matrix = MatrixInFile(M, N, dtype, data, dist)
            self.in_memory = False
        self.comm = None
        self.dtype = self.matrix.dtype

    def __len__(self):
        return len(self.matrix)

    def read_from_file(self):
        self.matrix = self.matrix.read(self.gd, self.myshape[1:])
        self.in_memory = True

    def multiply(self, alpha, opa, b, opb, beta, c, symmetric):
        self.matrix.multiply(alpha, opa, b.matrix, opb, beta, c, symmetric)
        if opa == 'N' and self.comm:
            if self.comm.size > 1:
                c.comm = self.comm
                c.state = 'a sum is needed'
            assert opb in 'TC' and b.comm is self.comm

    def matrix_elements(self, other, out=None, symmetric=False, cc=False):
        if out is None:
            out = Matrix(len(self), len(other), dtype=self.dtype)
        if isinstance(other, ArrayWaveFunctions):
            assert cc
            self.multiply(self.dv, 'N', other, 'C', 0.0, out, symmetric)
        else:
            assert not cc
            P_ani = {a: P_ni for a, P_ni in out.items()}
            other.integrate(self.array, P_ani, self.kpt)
        return out

    def add(self, lfc, coefs):
        lfc.add(self.array, dict(coefs.items()), self.kpt)

    def apply(self, func, out=None):
        out = out or self.new()
        func(self.array, out.array)
        return out

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
        ngpts = gd.n_c.prod()
        ArrayWaveFunctions.__init__(self, nbands, ngpts, dtype, data, dist)

        M = self.matrix

        if data is None:
            M.array = M.array.reshape(-1).reshape(M.dist.shape)

        self.myshape = (M.dist.shape[0],) + tuple(gd.n_c)
        self.gd = gd
        self.dv = gd.dv
        self.kpt = kpt
        self.spin = spin
        self.comm = gd.comm

    @property
    def array(self):
        if self.in_memory:
            return self.matrix.array.reshape(self.myshape)
        else:
            return self.matrix.array

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
        ng = ng0 = len(pd.Q_qG[kpt])
        if data is not None:
            assert ng == data.shape[1] or ng == data.length_of_last_dimension
            assert data.dtype == complex
        if dtype == float:
            ng *= 2
            if data is not None:
                data = data.view(float)
        ArrayWaveFunctions.__init__(self, nbands, ng, dtype, data, dist)
        self.pd = pd
        self.gd = pd.gd
        self.dv = pd.gd.dv / pd.gd.N_c.prod()
        self.kpt = kpt
        self.spin = spin
        self.myshape = (self.matrix.dist.shape[0], ng0)

    @property
    def array(self):
        if self.dtype == float and isinstance(self.matrix, Matrix):
            return self.matrix.array.view(complex)
        else:
            return self.matrix.array

    def matrix_elements(self, other, out=None, symmetric=False, cc=False):
        if isinstance(other, ArrayWaveFunctions):
            assert cc
            if out is None:
                out = Matrix(len(self), len(other), dtype=self.dtype)
            if self.dtype == complex:
                self.matrix.multiply(self.dv, 'N', other.matrix, 'C',
                                     0.0, out, symmetric)
            else:
                tmp_G = self.matrix.array[:, 0].copy()
                x = 0.5**0.5 if self is other else 0.5
                self.matrix.array[:, 0] *= x
                self.matrix.multiply(2 * self.dv, 'N', other.matrix, 'T',
                                     0.0, out, symmetric)
                self.matrix.array[:, 0] = tmp_G
        else:
            assert not cc
            P_ani = {a: P_ni for a, P_ni in out.items()}
            other.integrate(self.array, P_ani, self.kpt)
        return out

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
