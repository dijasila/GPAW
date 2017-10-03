import numpy as np
import scipy.linalg as linalg

import _gpaw
import gpaw.utilities.blas as blas


global_blacs_context_store = {}


def matrix(M):
    if isinstance(M, Matrix):
        return M
    return matrix(M.matrix)


def matrix_matrix_multiply(alpha, a, opa, b, opb, beta, c, symmetric=True):
    return matrix(a).multiply(alpha, opa, matrix(b), opb,
                              beta, c if c is None else matrix(c),
                              symmetric)


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

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if opa == 'C' and opb == 'T' and beta == 0.0:
            if c.array.dtype == float:
                blas.mmm(alpha, a.array, 'n', b.array, 't', 0.0, c.array)
            else:
                blas.mmm(alpha, b.array, 'n', a.array, 'c', 0.0, c.array)
                if symmetric:
                    np.negative(c.array.imag, c.array.imag)
                else:
                    c.array[:] = c.array.copy().T
        elif opa == 'H' and opb == 'N':
            blas.mmm(alpha, a.array, 'c', b.array, 'n', beta, c.array)
        elif opa == 'T' and opb == 'N':
            blas.mmm(alpha, a.array, 't', b.array, 'n', beta, c.array)
        elif opa == 'N' and opb == 'N':
            blas.mmm(alpha, a.array, 'n', b.array, 'n', beta, c.array)
        else:
            1 / 0

        if 0:
            if beta == 0.0:
                c2 = alpha * np.dot(op(a.array, opa), op(b.array, opb))
            else:
                assert beta == 1.0
                c2 = c.array + alpha * np.dot(op(a.array, opa),
                                              op(b.array, opb))

            if c2.size and abs(c.array - c2).max() > 0.000001:
                print(self, alpha, a, opa, b, opb, beta, c)
                print(c.array)
                print(c2)
                1 / 0

    def redist(self, M1, M2):
        M2.array[:] = M1.array

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
            if c == 1:
                br = (M + r - 1) // r
                bc = N
            elif r == 1:
                br = M
                bc = (N + c - 1) // c
            else:
                raise ValueError('Please specify block size!')
        else:
            br = bc = b

        n, m = _gpaw.get_blacs_local_shape(context, N, M, bc, br, 0, 0)
        self.shape = (m, n)
        lld = max(1, n)
        self.desc = np.array([1, context, N, M, bc, br, 0, 0, lld], np.intc)

    def __str__(self):
        return ('BLACSDistribution(global={}, local={}, blocksize={})'
                .format(*('{}x{}'.format(*shape)
                          for shape in [self.desc[3:1:-1],
                                        self.shape,
                                        self.desc[5:3:-1]])))

    def multiply(self, alpha, a, opa, b, opb, beta, destination, symmetric):
        print(alpha, a, opa, b, opb, beta, destination)

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

    def redist(self, M1, M2):
        _gpaw.scalapack_redist(self.desc, M2.desc,
                               M1.array, M2.array,
                               subN, subM,
                               ja + 1, ia + 1, jb + 1, ib + 1,  # 1-indexing
                               self.supercomm_bg.context, 'G')

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


class Matrix:
    def __init__(self, M, N, dtype=None, data=None, dist=None):
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
            self.array = np.empty(dist.shape, self.dtype)
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

    def multiply(self, alpha, opa, b, opb, beta=0.0, out=None,
                 symmetric=False):
        if out is None:
            out = Matrix()
        self.dist.multiply(alpha, self, opa, b, opb, beta, out, symmetric)
        return out

    def redist(self, other):
        self.dist.redist(self, other)

    def invcholesky(self):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array, 0)
        if self.comm is None or self.comm.rank == 0:
            self.dist.cholesky(self.array)
            self.dist.inv(self.array)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array, 0)
            self.state == 'fine'

    def eigh(self, eps_n):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array, 0)
        if self.comm is None or self.comm.rank == 0:
            self.dist.eigh(self.array, eps_n)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array, 0)
            self.comm.broadcast(eps_n, 0)
            self.state == 'fine'
