import numpy as np

import scipy.linalg as linalg
import gpaw.utilities.blas as blas
import _gpaw


global_blacs_context_store = {}


class NoDistribution:
    def __init__(self, M, N):
        self.shape = (M, N)

    def __str__(self):
        return 'NoDistribution({0}x{1})'.format(*self.shape)

    def mmm(self, alpha, a, opa, b, opb, beta, c):
        # print(self is b, self is b.source)
        if opa == 'C' and opb == 'T':
            assert not a.transposed and not b.transposed and c.transposed
            blas.mmm(alpha, b.a, 'n', a.a, 'c', beta, c.a.T)
        elif opa == 'T' and opb == 'N' and a.transposed:
            assert not b.transposed and not c.transposed
            blas.mmm(alpha, a.a.T, 'n', b.a, 'n', beta, c.a)
        else:
            assert not a.transposed and not b.transposed and c.transposed
            assert opa != 'C' and opb != 'C'
            blas.mmm(alpha, a.a, opa.lower(), b.a, opb.lower(), beta, c.a.T)

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

    def __str__(self):
        return ('BLACSDistribution(global={0}, local={1}, blocksize={2})'
                .format(*('{0}x{1}'.format(*shape)
                          for shape in [self.desc[2:4:1],
                                        self.shape,
                                        self.desc[4:6:1]])))

    def mmm(self, alpha, a, opa, b, opb, beta, destination):
        M, Ka = a.shape
        Kb, N = b.shape
        if opa == 'T':
            M, Ka = Ka, M
        if opb == 'T':
            Kb, N = N, Kb
        _gpaw.pblas_gemm(N, M, Ka, alpha, b.a, a.a,
                         beta, destination.a,
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
        self.dtype = np.dtype(dtype)

        if isinstance(dist, tuple):
            dist = create_distribution(M, N, *dist)
        self.dist = dist

        if data is None:
            self.a = np.empty(dist.shape, self.dtype).T
            self.transposed = True
        else:
            self.a = np.asarray(data).reshape(self.shape)
            self.transposed = False

        assert self.transposed == self.a.flags['F_CONTIGUOUS']

    def __str__(self):
        dist = str(self.dist).split('(')[1]
        return 'Matrix({0}: {1}'.format(self.dtype.name, dist)

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
            destination.a[:] = self.a
        else:
            assert beta == 1
            destination.a += self.a

    def __iadd__(self, x):
        x.eval(self, 1.0)
        return self

    def __mul__(self, x):
        if not isinstance(x, Product):
            x = (x, 'N')
        return Product((self, 'N'), x)

    def __rmul__(self, x):
        return Product(x, (self, 'N'))

    @property
    def T(self):
        return Product((self, 'T'))

    @property
    def C(self):
        return Product((self, 'C'))

    def mmm(self, alpha, opa, b, opb, beta, destination):
        if opa == 'C' and self.dtype == float:
            opa = 'N'
        self.dist.mmm(alpha, self, opa, b, opb, beta, destination)

    def cholesky(self):
        self.finish_sums()
        self.dist.cholesky(self.a)

    def inv(self):
        self.finish_sums()
        self.dist.inv(self.a)

    def eigh(self, eps_n):
        self.finish_sums()
        self.dist.eigh(self.a, eps_n)


class Product:
    def __init__(self, *x):
        self.things = []
        for p in x:
            if isinstance(p, Product):
                self.things.extend(p.things)
            else:
                self.things.append(p)

    def __str__(self):
        return str(self.things)

    def eval(self, destination, beta=0):
        if isinstance(self.things[0], (int, float)):
            alpha = self.things.pop(0)
        else:
            alpha = 1.0

        (a, opa), (b, opb) = self.things
        a.mmm(alpha, opa, b, opb, beta, destination)

    def __mul__(self, x):
        if isinstance(x, Matrix):
            x = Product((x, 'N'))
        return Product(self, x)


