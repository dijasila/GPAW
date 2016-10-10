import numpy as np

import scipy.linalg as linalg
import gpaw.utilities.blas as blas
import _gpaw


global_blacs_context_store = {}


class NoDistribution:
    def __init__(self, M, N):
        self.shape = (M, N)

    def mmm(self, alpha, a, opa, b, opb, beta, c):
        # print(self is b, self is b.source)
        if opa == 'C' and opb == 'T':
            assert not a.transposed and not b.transposed and c.transposed
            blas.mmm(alpha, b.x, 'n', a.x, 'c', beta, c.x.T)
        elif opa == 'T' and opb == 'N' and a.transposed:
            assert not b.transposed and not c.transposed
            blas.mmm(alpha, a.x.T, 'n', b.x, 'n', beta, c.x)
        else:
            assert not a.transposed and not b.transposed and c.transposed
            assert opa != 'C' and opb != 'C'
            blas.mmm(alpha, a.x, opa.lower(), b.x, opb.lower(), beta, c.x.T)

    def cholesky(self, S_nn):
        S_nn[:] = linalg.cholesky(S_nn)

    def inv(self, S_nn):
        S_nn[:] = linalg.inv(S_nn)

    def eigh(self, H_nn, eps_n):
        eps_n[:], H_nn[:] = linalg.eigh(H_nn)


class BLACSDistribution:
    def __init__(self, M, N, comm, r, c, b):
        key = (comm, r, c)
        context = global_blacs_context_store.get(key)
        if context is None:
            context = _gpaw.new_blacs_context(comm.get_c_object(), c, r, 'R')
            global_blacs_context_store[key] = context

        if b is None:
            assert c == 1
            br = (M + r - 1) // r
            bc = N
        else:
            br = bc = b

        n, m = _gpaw.get_blacs_local_shape(context, N, M, bc, br, 0, 0)
        self.shape = (m, n)
        lld = max(1, n)
        self.desc = np.array([1, context, N, M, bc, br, 0, 0, lld], np.intc)

    def mmm(self, alpha, a, opa, b, opb, beta, destination):
        M, Ka = a.shape
        Kb, N = b.shape
        if opa == 'T':
            M, Ka = Ka, M
        if opb == 'T':
            Kb, N = N, Kb
        _gpaw.pblas_gemm(N, M, Ka, alpha, b.x, a.x,
                         beta, destination.x,
                         b.dist.desc, a.dist.desc, destination.dist.desc,
                         opb, opa)

    def cholesky(self, S_nn):
        lapack.cholesky(S_nn)

    def inverse_cholesky(self, S_nn):
        lapack.inv(S_nn)

    def diagonalize(self, H_nn, eps_n):
        lapack.diagonalize(H_nn, eps_n)


def create_distribution(M, N, comm=None, r=1, c=1, b=None):
    if r == c == 1:
        return NoDistribution(M, N)
    return BLACSDistribution(M, N, comm, r, c, b)


class Matrix:
    def __init__(self, M, N, dtype=None, data=None, dist=None):
        self.shape = (M, N)

        if dtype is None:
            if data is None:
                dtype = float
            else:
                dtype = data.dtype
        self.dtype = dtype

        if isinstance(dist, tuple):
            dist = create_distribution(M, N, *dist)
        self.dist = dist

        if data is None:
            self.data = np.empty(dist.shape, self.dtype).T
            self.transposed = True
        else:
            self.data = np.asarray(data).reshape(self.shape)
            self.transposed = False

        self.x = self.data

    def __str__(self):
        return str(self.data)

    def new(self):
        return Matrix(*self.shape, dtype=self.dtype, dist=self.dist)

    def finish_sums(self):
        pass

    def __setitem__(self, i, x):
        # assert i == slice(None)
        x.eval(self)

    def eval(self, destination, beta=0):
        assert destination.dist == self.dist
        if beta == 0:
            destination.x[:] = self.x
        else:
            assert beta == 1
            destination.x += self.x

    def __iadd__(self, x):
        x.eval(self, 1.0)
        return self

    def __mul__(self, x):
        if not isinstance(x, Product):
            x = ('N', x)
        return Product(('N', self), x)

    def __rmul__(self, x):
        return Product(x, ('N', self))

    @property
    def T(self):
        return Product(('T', self))

    @property
    def C(self):
        return Product(('C', self))

    def mmm(self, alpha, opa, b, opb, beta, destination):
        if opa == 'C' and self.dtype == float:
            opa = 'N'
        self.dist.mmm(alpha, self, opa, b, opb, beta, destination)

    def cholesky(self):
        self.finish_sums()
        self.dist.cholesky(self.x)

    def inv(self):
        self.finish_sums()
        self.dist.inv(self.x)

    def eigh(self, eps_n):
        self.finish_sums()
        self.dist.eigh(self.x, eps_n)


class Product:
    def __init__(self, x, y=None):
        if y is None:
            self.things = [x]
        elif isinstance(y, Product):
            if isinstance(x, Product):
                self.things = x.things + y.things
            else:
                self.things = [x] + y.things
        else:
            self.things = [x, y]

    def __str__(self):
        return str(self.things)

    def eval(self, destination, beta=0):
        if isinstance(self.things[0], (int, float)):
            alpha = self.things.pop(0)
        else:
            alpha = 1.0

        (opa, a), (opb, b) = self.things
        a.mmm(alpha, opa, b, opb, beta, destination)

    def __mul__(self, x):
        if isinstance(x, Matrix):
            x = Product(('N', x))
        return Product(self, x)


