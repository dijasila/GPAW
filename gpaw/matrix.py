import numpy as np
import scipy.linalg as linalg

import _gpaw
from gpaw import debug
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
    serial = True

    def __init__(self, M, N):
        self.shape = (M, N)

    def __str__(self):
        return 'NoDistribution({}x{})'.format(*self.shape)

    def global_index(self, n):
        return n

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        blas.mmm(alpha, a.array, opa.lower(), b.array, opb.lower(),
                 beta, c.array)

    def invcholesky(self, S):
        L_nn = linalg.cholesky(S.array, lower=True, overwrite_a=True,
                               check_finite=debug)
        S.array[:] = linalg.inv(L_nn, overwrite_a=True, check_finite=debug)

    def eigh(self, H, cc, slcomm, rows, columns, blocksize):
        assert rows == columns == 1
        if cc and H.dtype == complex:
            np.negative(H.array.imag, H.array.imag)
        eps, H.array[:] = linalg.eigh(H.array,
                                      lower=True,  # ???
                                      overwrite_a=True,
                                      check_finite=debug)
        return eps


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
        # print(alpha, a, opa, b, opb, beta, c)
        # print(a.dist.desc, b.dist.desc, c.dist.desc)
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

    def eigh(self, H, cc, slcomm, rows, columns, blocksize):
        dist = (slcomm or self.comm, rows, columns, blocksize)
        H0 = H.new(dist=dist)
        H.redist(H0)
        eps = np.empty(H.shape[0])
        if rows == columns == 1:
            if self.comm.rank == 0:
                eps = NoDistribution.eigh('self', H0, cc,
                                          None, 1, 1, None)
            self.comm.broadcast(eps, 0)
        else:
            if cc and H.dtype == complex:
                array = H0.array.conj()
            else:
                array = H0.array.copy()
            info = _gpaw.scalapack_diagonalize_dc(array, H0.dist.desc, 'U',
                                                  H0.array, eps)
            assert info == 0, info

        H0.redist(H)
        return eps


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

        self.comm = None
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

    def redist(self, other,c=None):
        if self is other:
            return
        d1 = self.dist
        d2 = other.dist
        if isinstance(d1, NoDistribution):
            if isinstance(d2, NoDistribution):
                other.array[:] = self.array
            else:
                M, N = self.shape
                dist = create_distribution(M, N, other.dist.comm, 1, 1)
                redist(dist, self.array, other.dist, other.array,
                       other.dist.desc[1])
        else:
            if isinstance(d2, NoDistribution):
                M, N = self.shape
                d2 = create_distribution(M, N, d1.comm, 1, 1)
                redist(d1, self.array, d2, other.array,
                       d2.desc[1])
                return
            if d2.comm.size > d1.comm.size:
                M, N = self.shape
                d1 = create_distribution(M, N, d2.comm, d1.rows, d1.columns, d1.blocksize)
                redist(d1, self.array, d2, other.array,
                       d2.desc[1])
                return
            if d2.comm.size < d1.comm.size:
                M, N = self.shape
                d2 = create_distribution(M, N, d1.comm, d2.rows, d2.columns, d2.blocksize)
                redist(d1, self.array, d2, other.array,
                       d1.desc[1])
                return
            ctx = min((d[4] * d[5], d[1])
                      for d in [self.dist.desc, other.dist.desc])[1]
            redist(self.dist, self.array, other.dist, other.array, ctx)

    def invcholesky(self):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array, 0)
        if self.comm is None or self.comm.rank == 0:
            self.dist.invcholesky(self)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array, 0)
            self.state == 'everything is fine'

    def eigh(self, cc=False, scalapack=(None, 1, 1, None)):
        if self.state == 'a sum is needed':
            self.comm.sum(self.array, 0)

        slcomm, rows, columns, blocksize = scalapack
        if rows == columns == 1 and self.comm is not None and self.comm.rank > 0:
            eps_n = np.empty(self.shape[0])
        else:
            eps_n = self.dist.eigh(self, cc, slcomm, rows, columns, blocksize)

        assert (self.state == 'a sum is needed') == (self.comm is not None and self.comm.size > 1)
        if self.comm is not None and self.comm.size > 1:
            self.comm.broadcast(self.array, 0)
            self.comm.broadcast(eps_n, 0)
            self.state == 'everything is fine'
        return eps_n

    def complex_conjugate(self):
        if self.dtype == complex:
            np.negative(self.array.imag, self.array.imag)
