import numpy as np
import scipy.linalg as linalg

import _gpaw
import gpaw.utilities.blas as blas


global_blacs_context_store = {}


def op(a, opa):
    if opa == 'N':
        return a
    if opa == 'C':
        return a.conj()
    if opa == 'T':
        return a.T
    return a.T.conj()


class NoDistribution:
    serial = True

    def __init__(self, M, N):
        self.shape = (M, N)

    def __str__(self):
        return 'NoDistribution({}x{})'.format(*self.shape)

    def global_index(self, n):
        return n

    def multiply(self, alpha, a, opa, b, opb, beta, c):
        if beta == 0.0:
            c2 = alpha * np.dot(op(a.array, opa), op(b.array, opb))
        else:
            assert beta == 1.0
            c2 = c.array + alpha * np.dot(op(a.array, opa), op(b.array, opb))
        c.array[:] = c2
        return

        # print(self is b, self is b.source)
        print('hej')
        if opa == 'C' and opb == 'T':
            assert not a.transposed and not b.transposed and c.transposed
            blas.mmm(alpha, b.array, 'n', a.array, 'c', beta, c.array.T, 'n')
        elif opa == 'T' and opb == 'N' and a.transposed:
            assert not b.transposed and not c.transposed
            blas.mmm(alpha, a.array.T, 'n', b.array, 'n', beta, c.array)
        else:
            assert not a.transposed and not b.transposed and c.transposed
            assert opa != 'C' and opb != 'C'
            print(c.array)
            blas.mmm(alpha, a.array, opa.lower(), b.array, opb.lower(), beta,
                     c.array, 'n')  # .T)
        if abs(c.array - c2).max() > 0.000001:
            print(self, alpha, a, opa, b, opb, beta, c)
            print(a.transposed, b.transposed, c.transposed)
            print(c.array)
            print(c2)
            print(np.dot(a.array[0], b.array[1]))
            1 / 0
        c.array[:] = c2

    def cholesky(self, S_nn):
        S_nn[:] = linalg.cholesky(S_nn)

    def inv(self, S_nn):
        S_nn[:] = linalg.inv(S_nn)

    def eigh(self, H_nn, eps_n):
        eps_n[:], H_nn[:] = linalg.eigh(H_nn)


class BLACSDistribution:
    serial = False

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
        return ('BLACSDistribution(global={}, local={}, blocksize={})'
                .format(*('{}x{}'.format(*shape)
                          for shape in [self.desc[2:4],
                                        self.shape,
                                        self.desc[4:6]])))

    def multiply(self, alpha, a, opa, b, opb, beta, destination):
        M, Ka = a.shape
        Kb, N = b.shape
        if opa == 'T':
            M, Ka = Ka, M
        if opb == 'T':
            Kb, N = N, Kb
        _gpaw.pblas_gemm(N, M, Ka, alpha, b.array, a.array,
                         beta, destination.array,
                         b.dist.desc, a.dist.desc, destination.dist.desc,
                         opb, opa)

    def cholesky(self, S_nn):
        1 / 0  # 1 / 0  # lapack.cholesky(S_nn)

    def inverse_cholesky(self, S_nn):
        1 / 0  # lapack.inv(S_nn)

    def diagonalize(self, H_nn, eps_n):
        1 / 0  # lapack.diagonalize(H_nn, eps_n)


def create_distribution(M, N, comm=None, r=1, c=1, b=None):
    if comm is None or comm.size == 1:
        assert r == 1 and abs(c) == 1 or c == 1 and abs(r) == 1
        return NoDistribution(M, N)

    return BLACSDistribution(M, N, comm, r, c, b)


class Op:
    def __init__(self, M, op='N'):
        if isinstance(M, Op):
            assert op == 'N'
            M = M.M
            op = M.op
        elif hasattr(M, 'matrix'):
            M = M.matrix
        self.M = M
        self.op = op

    def __mul__(self, other):
        if not isinstance(other, Op):
            other = Op(other)
        return Product(self, other)


class Matrix:
    def __init__(self, M, N, dtype=None, data=None, dist=None, order='F'):
        self.shape = (M, N)

        if dtype is None:
            if data is None:
                dtype = float
            else:
                dtype = data.dtype
        self.dtype = np.dtype(dtype)

        dist = dist or ()
        if isinstance(dist, tuple):
            dist = create_distribution(M, N, *dist)
        self.dist = dist

        if data is None:
            self.array = np.empty(dist.shape, self.dtype, order=order)
        else:
            self.array = data.reshape(dist.shape)

        self.comm = None
        self.state = 'fine'

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        dist = str(self.dist).split('(')[1]
        return 'Matrix({}: {}'.format(self.dtype.name, dist)

    def new(self):
        return Matrix(*self.shape, dtype=self.dtype, dist=self.dist)

    def __setitem__(self, i, x):
        # assert i == slice(None)
        if isinstance(x, np.ndarray):
            1 / 0  # sssssself.array[:] = x
        else:
            x.eval(self)

    def __iadd__(self, x):
        x.eval(self, 1.0)
        return self

    def __mul__(self, other):
        return Product(Op(self), Op(other))

    def __rmul__(self, other):
        return Product(Op(other), Op(self))

    @property
    def T(self):
        return Op(self, 'T')

    @property
    def C(self):
        return Op(self, 'C')

    @property
    def H(self):
        return Op(self, 'H')

    def multiply(self, alpha, opa, b, opb, beta, out):
        if opa == 'Ccccccccccccccccccccccccc' and self.dtype == float:
            opa = 'N'
        if out is None:
            if opa in 'NC':
                M = self.shape[0]
            else:
                M = self.shape[1]
            if opb in 'NC':
                N = b.shape[1]
            else:
                N = b.shape[0]
            out = Matrix(M, N)

        self.dist.multiply(alpha, self, opa, b, opb, beta, out)
        return out

    def invcholesky(self):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array.T, 0)
        if self.comm is None or self.comm.rank == 0:
            self.dist.cholesky(self.array)
            self.dist.inv(self.array)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array.T, 0)
            self.state == 'fine'

    def eigh(self, eps_n):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array.T, 0)
        if self.comm is None or self.comm.rank == 0:
            self.dist.eigh(self.array, eps_n)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array.T, 0)
            self.comm.broadcast(eps_n, 0)
            self.state == 'fine'


class Product:
    def __init__(self, a, b):
        self.array = a
        self.b = b

    def __str__(self):
        return str(self.things)

    def eval(self, out=None, beta=0.0, alpha=1.0):
        a = self.array
        b = self.b
        return a.M.multiply(alpha, a.op, b.M, b.op, beta, out)

    def integrate(self, out=None, hermetian=False):
        a = self.array
        b = self.b
        assert a.op == 'C' or a.M.dtype == float and a.op == 'N'
        assert b.op == 'N'
        return a.M.integrate(b.M, out, hermetian)
