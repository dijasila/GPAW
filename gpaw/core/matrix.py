"""BLACS distributed matrix object."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import scipy.linalg as linalg

import _gpaw
from gpaw import debug
from gpaw.mpi import serial_comm, _Communicator
import gpaw.utilities.blas as blas


_global_blacs_context_store: Dict[Tuple[_Communicator, int, int], int] = {}


def suggest_blocking(N, ncpus):
    """Suggest blocking of NxN matrix.

    Returns rows, columns, blocksize tuple.

    >>> suggest_blocking(10, 6)
    (3, 2, 2)
    """

    nprow = ncpus
    npcol = 1

    # Make npcol and nprow as close to each other as possible
    npcol_try = npcol
    while npcol_try < nprow:
        if ncpus % npcol_try == 0:
            npcol = npcol_try
            nprow = ncpus // npcol
        npcol_try += 1

    assert npcol * nprow == ncpus

    # ScaLAPACK creates trouble if there aren't at least a few whole blocks.
    # Choose block size so that there will always be at least one whole block
    # and at least two blocks in total.
    blocksize = max((N - 2) // max(nprow, npcol), 1)
    # The next commented line would give more whole blocks.
    # blocksize = max(N // max(nprow, npcol) - 2, 1)

    # Use block size that is a power of 2 and at most 64
    blocksize = 2**int(np.log2(blocksize))
    blocksize = max(min(blocksize, 64), 1)

    return nprow, npcol, blocksize


class Matrix:
    def __init__(self, M, N, dtype=None, data=None, dist=None):
        """Matrix object.

        M: int
            Rows.
        N: int
            Columns.
        dtype: type
            Data type (float or complex).
        dist: tuple or None
            BLACS distribution given as (communicator, rows, colums, blocksize)
            tuple.  Default is None meaning no distribution.
        data: ndarray or None.
            Numpy ndarray to use for starage.  By default, a new ndarray
            will be allocated.
            """
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
            self.data = np.empty(dist.shape, self.dtype)
        else:
            self.data = data.reshape(dist.shape)

    def __repr__(self):
        dist = str(self.dist).split('(')[1]
        return 'Matrix({}: {}'.format(self.dtype.name, dist)

    def new(self, dist='inherit', data=None):
        """Create new matrix of same shape and dtype.

        Default is to use same BLACS distribution.  Use dist to use another
        distribution.
        """
        return Matrix(*self.shape,
                      dtype=self.dtype,
                      dist=self.dist if dist == 'inherit' else dist,
                      data=data)

    def __setitem__(self, item, value):
        assert item == slice(None)
        assert isinstance(value, Matrix)
        self.data[:] = value.data

    def multiply(self,
                 other,
                 alpha=1.0,
                 opa='N',
                 opb='N',
                 out=None,
                 beta=0.0,
                 symmetric=False) -> Matrix:
        if not isinstance(other, Matrix):
            other = other.matrix
        A = self
        B = other
        dist = self.dist
        if out is None:
            assert beta == 0.0
            M = A.shape[0] if opa == 'N' else A.shape[1]
            N = B.shape[1] if opb == 'N' else B.shape[0]
            out = Matrix(M, N, A.dtype,
                         dist=(dist.comm, dist.rows, dist.columns))
        elif not isinstance(out, Matrix):
            out = out.matrix

        if dist.comm.size > 1:
            # Special cases that don't need scalapack - most likely also
            # faster:
            if alpha == 1.0 and opa == 'N' and opb == 'N':
                return fastmmm(A, B, out, beta)
            if alpha == 1.0 and beta == 1.0 and opa == 'N' and opb == 'C':
                if symmetric:
                    return fastmmm2(A, B, out)
                else:
                    return fastmmm2notsym(A, B, out)

        dist.multiply(alpha, A, opa, B, opb, beta, out, symmetric)
        return out

    def redist(self, other):
        """Redistribute to other BLACS layout."""
        if self is other:
            return
        d1 = self.dist
        d2 = other.dist
        n1 = d1.rows * d1.columns
        n2 = d2.rows * d2.columns
        if n1 == n2 == 1:
            other.data[:] = self.data
            return

        if n2 == 1 and d1.blocksize is None:
            assert d2.blocksize is None
            comm = d1.comm
            if comm.rank == 0:
                M = len(self)
                m = (M + comm.size - 1) // comm.size
                other.data[:m] = self.data
                for r in range(1, comm.size):
                    m1 = min(r * m, M)
                    m2 = min(m1 + m, M)
                    comm.receive(other.data[m1:m2], r)
            else:
                comm.send(self.data, 0)
            return

        if n1 == 1 and d2.blocksize is None:
            assert d1.blocksize is None
            comm = d1.comm
            if comm.rank == 0:
                M = len(self)
                m = (M + comm.size - 1) // comm.size
                other.data[:] = self.data[:m]
                for r in range(1, comm.size):
                    m1 = min(r * m, M)
                    m2 = min(m1 + m, M)
                    comm.send(self.data[m1:m2], r)
            else:
                comm.receive(other.data, 0)
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
            redist(d1, self.data, d2, other.data, ctx)

    def invcholesky(self):
        """Inverse of Cholesky decomposition.

        Only the lower part is used.
        """
        if self.dist.comm.size > 1:
            S = self.new(dist=(self.dist.comm, 1, 1))
            self.redist(S)
        else:
            S = self
        if self.dist.comm.rank == 0:
            if debug:
                S.data[np.triu_indices(S.shape[0], 1)] = 42.0
            L_nn = linalg.cholesky(S.data,
                                   lower=True,
                                   overwrite_a=True,
                                   check_finite=debug)
            S.data[:] = linalg.inv(L_nn,
                                   overwrite_a=True,
                                   check_finite=debug)
        if S is not self:
            S.redist(self)

    def eigh(self, cc=False, scalapack=(None, 1, 1, None)):
        """Calculate eigenvectors and eigenvalues.

        Matrix must be symmetric/hermitian and stored in lower half.

        cc: bool
            Complex conjugate matrix before finding eigenvalues.
        scalapack: tuple
            BLACS distribution for ScaLapack to use.  Default is to do serial
            diagonalization.
        """
        slcomm, rows, columns, blocksize = scalapack

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
            if self.dist.comm.rank == 0:
                if cc and H.dtype == complex:
                    np.negative(H.data.imag, H.data.imag)
                if debug:
                    H.data[np.triu_indices(H.shape[0], 1)] = 42.0
                eps[:], H.data.T[:] = linalg.eigh(H.data,
                                                  lower=True,  # ???
                                                  overwrite_a=True,
                                                  check_finite=debug)
            self.dist.comm.broadcast(eps, 0)
        else:
            if slcomm.rank < rows * columns:
                assert cc
                array = H.data.copy()
                info = _gpaw.scalapack_diagonalize_dc(array, H.dist.desc, 'U',
                                                      H.data, eps)
                assert info == 0, info

            # necessary to broadcast eps when some ranks are not used
            # in current scalapack parameter set
            # eg. (2, 1, 2) with 4 processes
            if rows * columns < slcomm.size:
                H.dist.comm.broadcast(eps, 0)

        if redist:
            H.redist(self)

        return eps

    def complex_conjugate(self):
        """Inplace complex conjugation."""
        if self.dtype == complex:
            np.negative(self.data.imag, self.data.imag)


def _matrix(M):
    """Dig out Matrix object from wrapper(s)."""
    if isinstance(M, Matrix):
        return M
    return _matrix(M.matrix)


class MatrixDistribution:
    def matrix(self, dtype=None, data=None):
        return Matrix(*self.full_shape, dtype=dtype, data=data, dist=self)


class NoDistribution(MatrixDistribution):
    comm = serial_comm
    rows = 1
    columns = 1
    blocksize = None

    def __init__(self, M, N):
        self.shape = (M, N)
        self.full_shape = (M, N)

    def __str__(self):
        return 'NoDistribution({}x{})'.format(*self.shape)

    def global_index(self, n):
        return n

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if symmetric:
            if opa == 'N':
                assert opb == 'C' or opb == 'T' and a.dtype == float
                if a is b:
                    blas.rk(alpha, a.data, beta, c.data)
                else:
                    if beta == 1.0 and a.shape[1] == 0:
                        return
                    blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data)
            else:
                assert opa == 'C' and opb == 'N'
                assert a is not b
                blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data, 'n')

        else:
            blas.mmm(alpha, a.data, opa, b.data, opb, beta, c.data)


class BLACSDistribution(MatrixDistribution):
    serial = False

    def __init__(self, M, N, comm, r, c, b):
        self.comm = comm
        self.rows = r
        self.columns = c
        self.blocksize = b
        self.full_shape = (M, N)

        key = (comm, r, c)
        context = _global_blacs_context_store.get(key)
        if context is None:
            try:
                context = _gpaw.new_blacs_context(comm.get_c_object(),
                                                  c, r, 'R')
            except AttributeError:
                pass
            else:
                _global_blacs_context_store[key] = context

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

        if context is None:
            assert b is None
            assert c == 1
            n = N
            m = min((comm.rank + 1) * br, M) - min(comm.rank * br, M)
        else:
            n, m = _gpaw.get_blacs_local_shape(context, N, M, bc, br, 0, 0)
        if n < 0 or m < 0:
            n = m = 0
        self.shape = (m, n)
        lld = max(1, n)
        if context is not None:
            self.desc = np.array([1, context, N, M, bc, br, 0, 0, lld],
                                 np.intc)

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
            assert opa == 'N'
            assert opb == 'C' or opb == 'T' and a.dtype == float
            N, K = a.shape
            if a is b:
                _gpaw.pblas_rk(N, K, alpha, a.data,
                               beta, c.data,
                               a.dist.desc, c.dist.desc,
                               'U')
            else:
                _gpaw.pblas_r2k(N, K, 0.5 * alpha, b.data, a.data,
                                beta, c.data,
                                b.dist.desc, a.dist.desc, c.dist.desc,
                                'U')
        else:
            Ka, M = a.shape
            N, Kb = b.shape
            if opa == 'N':
                Ka, M = M, Ka
            if opb == 'N':
                N, Kb = Kb, N
            _gpaw.pblas_gemm(N, M, Ka, alpha, b.data, a.data,
                             beta, c.data,
                             b.dist.desc, a.dist.desc, c.dist.desc,
                             opb, opa)


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


def fastmmm(m1, m2, m3, beta):
    comm = m1.dist.comm

    buf1 = m2.data

    N = len(m1)
    n = (N + comm.size - 1) // comm.size

    for r in range(comm.size):
        if r == 0:
            buf2 = np.empty((n, buf1.shape[1]), dtype=buf1.dtype)

        rrequest = None
        srequest = None
        if r < comm.size - 1:
            rrank = (comm.rank + r + 1) % comm.size
            rn1 = min(rrank * n, N)
            rn2 = min(rn1 + n, N)
            if rn2 > rn1:
                rrequest = comm.receive(buf2[:rn2 - rn1], rrank, 21, False)
            srank = (comm.rank - r - 1) % comm.size
            if len(m2.data) > 0:
                srequest = comm.send(m2.data, srank, 21, False)

        r0 = (comm.rank + r) % comm.size
        n1 = min(r0 * n, N)
        n2 = min(n1 + n, N)
        blas.mmm(1.0, m1.data[:, n1:n2], 'N', buf1[:n2 - n1], 'N',
                 beta, m3.data)

        beta = 1.0

        if r == 0:
            buf1 = np.empty_like(buf2)

        buf1, buf2 = buf2, buf1

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

    return m3


def fastmmm2(a, b, out):
    if a.comm:
        assert b.comm is a.comm
        if a.comm.size > 1:
            assert out.comm == a.comm
            assert out.state == 'a sum is needed'

    comm = a.dist.comm
    M, N = a.shape
    m = (M + comm.size - 1) // comm.size
    mym = len(a.data)

    buf1 = np.empty((m, N), dtype=a.dtype)
    buf2 = np.empty((m, N), dtype=a.dtype)
    half = comm.size // 2
    aa = a.data
    bb = b.data

    for r in range(half + 1):
        rrequest = None
        srequest = None

        if r < half:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            skip = (comm.size % 2 == 0 and r == half - 1)
            m1 = min(rrank * m, M)
            m2 = min(m1 + m, M)
            if not (skip and comm.rank < half) and m2 > m1:
                rrequest = comm.receive(buf1[:m2 - m1], rrank, 11, False)
            if not (skip and comm.rank >= half) and mym > 0:
                srequest = comm.send(b.data, srank, 11, False)

        if not (comm.size % 2 == 0 and r == half and comm.rank < half):
            m1 = min(((comm.rank - r) % comm.size) * m, M)
            m2 = min(m1 + m, M)
            if r == 0:
                # symmmmmmmmmmmmmmmmmmmmmmetricccccccccccccccc
                blas.mmm(1.0, aa, 'N', bb, 'C', 1.0, out.data[:, m1:m2])
            else:
                beta = 1.0 if r <= comm.rank else 0.0
                blas.mmm(1.0, aa, 'N', buf2[:m2 - m1], 'C',
                         beta, out.data[:, m1:m2])
            # out.data[:, m1:m2] = m12.data[:, :m2 - m1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        bb = buf1
        buf1, buf2 = buf2, buf1

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                m1 = min(column * m, M)
                m2 = min(m1 + m, M)
                if mym > 0 and m2 > m1:
                    requests.append(
                        comm.send(out.data[:, m1:m2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                m1 = min(row * m, M)
                m2 = min(m1 + m, M)
                if mym > 0 and m2 > m1:
                    block = np.empty((mym, m2 - m1), out.dtype)
                    blocks.append((m1, m2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for m1, m2, block in blocks:
        out.data[:, m1:m2] += block

    return out


def fastmmm2notsym(a, b, out):
    if a.comm:
        assert b.comm is a.comm
        if a.comm.size > 1:
            assert out.comm == a.comm
            assert out.state == 'a sum is needed'

    comm = a.dist.comm
    M, N = a.shape
    m = (M + comm.size - 1) // comm.size
    mym = len(a.data)

    buf1 = np.empty((m, N), dtype=a.dtype)
    buf2 = np.empty((m, N), dtype=a.dtype)
    aa = a.data
    bb = b.data

    for r in range(comm.size):
        rrequest = None
        srequest = None

        if r < comm.size - 1:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            m1 = min(rrank * m, M)
            m2 = min(m1 + m, M)
            if m2 > m1:
                rrequest = comm.receive(buf1[:m2 - m1], rrank, 11, False)
            if mym > 0:
                srequest = comm.send(b.data, srank, 11, False)

        m1 = min(((comm.rank - r) % comm.size) * m, M)
        m2 = min(m1 + m, M)
        # symmmmmmmmmmmmmmmmmmmmmmetricccccccccccccccc ??
        blas.mmm(1.0, aa, 'N', bb[:m2 - m1], 'C', 1.0, out.data[:, m1:m2])

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        bb = buf1
        buf1, buf2 = buf2, buf1

    return out
