import numpy as np
import scipy.linalg as linalg

import _gpaw
from gpaw import debug
from gpaw.mpi import serial_comm
import gpaw.utilities.blas as blas


global_blacs_context_store = {}


def matrix(M):
    if isinstance(M, Matrix):
        return M
    return matrix(M.matrix)


def matrix_matrix_multiply(alpha, a, opa, b, opb, beta, c, symmetric=False):
    return matrix(a).multiply(alpha, opa, matrix(b), opb,
                              beta, c if c is None else matrix(c),
                              symmetric)


def suggest_blocking(N, ncpus):
    """Suggest blocking of NxN matrix."""

    nprow = ncpus
    npcol = 1

    # Get a sort of reasonable number of columns/rows
    while npcol < nprow and nprow % 2 == 0:
        npcol *= 2
        nprow //= 2

    assert npcol * nprow == ncpus

    # ScaLAPACK creates trouble if there aren't at least a few
    # whole blocks; choose block size so there will always be
    # several blocks.  This will crash for small test systems,
    # but so will ScaLAPACK in any case
    blocksize = min(-(-N // 4), 64)

    return nprow, npcol, blocksize


class NoDistribution:
    comm = serial_comm
    rows = 1
    columns = 1
    blocksize = None

    def __init__(self, M, N):
        self.shape = (M, N)

    def __str__(self):
        return 'NoDistribution({}x{})'.format(*self.shape)

    def global_index(self, n):
        return n

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if symmetric:
            assert opa == 'N'
            assert opb == 'C' or opb == 'T' and a.dtype == float
            if a is b:
                blas.rk(alpha, a.array, beta, c.array)
            else:
                if beta == 1.0 and a.shape[1] == 0:
                    return
                blas.r2k(0.5 * alpha, a.array, b.array, beta, c.array)
        else:
            blas.mmm(alpha, a.array, opa.lower(), b.array, opb.lower(),
                     beta, c.array)

    def invcholesky(self, S):
        if debug:
            S.array[np.triu_indices(S.shape[0], 1)] = 42.0
        L_nn = linalg.cholesky(S.array, lower=True, overwrite_a=True,
                               check_finite=debug)
        S.array[:] = linalg.inv(L_nn, overwrite_a=True, check_finite=debug)


class BLACSDistribution:
    serial = False

    def __init__(self, M, N, comm, r, c, b):
        self.comm = comm
        self.rows = r
        self.columns = c
        self.blocksize = b

        key = (comm, r, c)
        context = global_blacs_context_store.get(key)
        if context is None:
            context = _gpaw.new_blacs_context(comm.get_c_object(), c, r, 'R')
            global_blacs_context_store[key] = context

        if b is None:
            if c == 1:
                br = (M + r - 1) // r
                bc = max(1, N)
            elif r == 1:
                br = M
                bc = (N + c - 1) // c
            else:
                raise ValueError('Please specify block size!')
        else:
            br = bc = b

        n, m = _gpaw.get_blacs_local_shape(context, N, M, bc, br, 0, 0)
        if n < 0 or m < 0:
            n = m = 0
        self.shape = (m, n)
        lld = max(1, n)
        self.desc = np.array([1, context, N, M, bc, br, 0, 0, lld], np.intc)

    def __str__(self):
        return ('BLACSDistribution(global={}, local={}, blocksize={})'
                .format(*('{}x{}'.format(*shape)
                          for shape in [self.desc[3:1:-1],
                                        self.shape,
                                        self.desc[5:3:-1]])))

    def global_index(self, myi):
        return self.comm.rank * int(self.desc[5]) + myi

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if symmetric:
            assert opa == 'N' and opb == 'C'
            N, K = a.shape
            if a is b:
                _gpaw.pblas_rk(N, K, 0.5 * alpha, a.array,
                               beta, c.array,
                               a.dist.desc, c.dist.desc,
                               'U')
            else:
                _gpaw.pblas_r2k(N, K, alpha, b.array, a.array,
                                beta, c.array,
                                b.dist.desc, a.dist.desc, c.dist.desc,
                                'U')
        else:
            Ka, M = a.shape
            N, Kb = b.shape
            if opa == 'N':
                Ka, M = M, Ka
            if opb == 'N':
                N, Kb = Kb, N
            _gpaw.pblas_gemm(N, M, Ka, alpha, b.array, a.array,
                             beta, c.array,
                             b.dist.desc, a.dist.desc, c.dist.desc,
                             opb, opa)

    def invcholesky(self, S):
        S0 = S.new(dist=(self.comm, 1, 1))
        S.redist(S0)
        if self.comm.rank == 0:
            NoDistribution.invcholesky('self', S0)
        S0.redist(S)


def redist(dist1, M1, dist2, M2, context):
    _gpaw.scalapack_redist(dist1.desc, dist2.desc,
                           M1, M2,
                           dist1.desc[2], dist1.desc[3],
                           1, 1, 1, 1,  # 1-indexing
                           context, 'G')


def create_distribution(M, N, comm=None, r=1, c=1, b=None):
    if comm is None or comm.size == 1:
        assert r == 1 and abs(c) == 1 or c == 1 and abs(r) == 1
        return NoDistribution(M, N)

    return BLACSDistribution(M, N, comm,
                             r if r != -1 else comm.size,
                             c if c != -1 else comm.size,
                             b)


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

        self.comm = serial_comm
        self.state = 'everything is fine'

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        dist = str(self.dist).split('(')[1]
        return 'Matrix({}: {}'.format(self.dtype.name, dist)

    def new(self, dist='inherit'):
        return Matrix(*self.shape, dtype=self.dtype,
                      dist=self.dist if dist == 'inherit' else dist)

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
        if self is other:
            return
        d1 = self.dist
        d2 = other.dist
        n1 = d1.rows * d1.columns
        n2 = d2.rows * d2.columns
        if n1 == n2 == 1:
            other.array[:] = self.array
            return
        c = d1.comm if d1.comm.size > d2.comm.size else d2.comm
        n = max(n1, n2)
        if n < c.size:
            c = c.new_communicator(np.arange(n))
        if c is not None:
            M, N = self.shape
            d1 = create_distribution(M, N, c,
                                     d1.rows, d1.columns, d1.blocksize)
            d2 = create_distribution(M, N, c,
                                     d2.rows, d2.columns, d2.blocksize)
            if n1 == n:
                ctx = d1.desc[1]
            else:
                ctx = d2.desc[1]
            redist(d1, self.array, d2, other.array, ctx)

    def invcholesky(self):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array, 0)
        if self.comm is None or self.comm.rank == 0:
            self.dist.invcholesky(self)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array, 0)
            self.state == 'everything is fine'

    def eigh(self, cc=False, scalapack=(None, 1, 1, None)):
        slcomm, rows, columns, blocksize = scalapack

        if self.state == 'a sum is needed':
            self.comm.sum(self.array, 0)

        slcomm = slcomm or self.dist.comm
        dist = (slcomm, rows, columns, blocksize)

        redist = (rows != self.dist.rows or
                  columns != self.dist.columns or
                  blocksize != self.dist.blocksize)

        if redist:
            H = self.new(dist=dist)
            self.redist(H)
        else:
            assert self.dist.comm.size == slcomm.size
            H = self

        eps = np.empty(H.shape[0])

        if rows * columns == 1:
            if self.comm.rank == 0 and self.dist.comm.rank == 0:
                if cc and H.dtype == complex:
                    np.negative(H.array.imag, H.array.imag)
                eps[:], H.array.T[:] = linalg.eigh(H.array,
                                                   lower=True,  # ???
                                                   overwrite_a=True,
                                                   check_finite=debug)
            self.dist.comm.broadcast(eps, 0)
        elif slcomm.rank < rows * columns:
            assert cc
            array = H.array.copy()
            info = _gpaw.scalapack_diagonalize_dc(array, H.dist.desc, 'U',
                                                  H.array, eps)
            assert info == 0, info

        if redist:
            H.redist(self)

        assert (self.state == 'a sum is needed') == (
            self.comm is not None and self.comm.size > 1)
        if self.comm.size > 1:
            self.comm.broadcast(self.array, 0)
            self.comm.broadcast(eps, 0)
            self.state == 'everything is fine'

        return eps

    def complex_conjugate(self):
        if self.dtype == complex:
            np.negative(self.array.imag, self.array.imag)
