import numpy as np

import scipy.linalg as linalg
import gpaw.utilities.blas as blas
import _gpaw


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

    def mmm(self, alpha, a, opa, b, opb, beta, c):
        if beta == 0.0:
            c2 = alpha * np.dot(op(a.a, opa), op(b.a, opb))
        else:
            assert beta == 1.0
            c2 = c.a + alpha * np.dot(op(a.a, opa), op(b.a, opb))
        c.a[:] = c2
        return
        #return
        # print(self is b, self is b.source)
        print('hej')
        if opa == 'C' and opb == 'T':
            assert not a.transposed and not b.transposed and c.transposed
            blas.mmm(alpha, b.a, 'n', a.a, 'c', beta, c.a.T, 'n')
        elif opa == 'T' and opb == 'N' and a.transposed:
            assert not b.transposed and not c.transposed
            blas.mmm(alpha, a.a.T, 'n', b.a, 'n', beta, c.a)
        else:
            assert not a.transposed and not b.transposed and c.transposed
            assert opa != 'C' and opb != 'C'
            print(c.a)
            blas.mmm(alpha, a.a, opa.lower(), b.a, opb.lower(), beta, c.a, 'n')#.T)
        if abs(c.a-c2).max() > 0.000001:
            print(self, alpha, a, opa, b, opb, beta, c)
            print(a.transposed, b.transposed, c.transposed)
            print(c.a)
            print(c2)
            print(np.dot(a.a[0], b.a[1]))
            1 / 0
        c.a[:] = c2

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
            self.a = np.empty(dist.shape, self.dtype, order='F')
        else:
            self.a = data

        self.array = self.a

        self.transposed = self.a.flags['F_CONTIGUOUS']

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        dist = str(self.dist).split('(')[1]
        return 'Matrix({}: {}'.format(self.dtype.name, dist)

    def new(self):
        return Matrix(*self.shape, dtype=self.dtype, dist=self.dist)

    def finish_sums(self):
        pass

    def __setitem__(self, i, x):
        # assert i == slice(None)
        if isinstance(x, np.ndarray):
            sssssself.array[:] = x
        else:
            x.eval(self)

    def __array_____(self, dtype):
        assert self.dtype == dtype
        assert self.dist.serial
        return self.a

    def evallllllllll(self, destination=None, beta=0):
        if destination is None:
            destination = ...
        assert destination.dist == self.dist
        if beta == 0:
            destination.a[:] = self.a
        else:
            assert beta == 1
            destination.a += self.a
        return destination

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

    def mmm(self, alpha, opa, b, opb, beta, out):
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
        self.dist.mmm(alpha, self, opa, b, opb, beta, out)
        return out

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
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return str(self.things)

    def eval(self, out=None, beta=0.0, alpha=1.0):
        a = self.a
        b = self.b
        return a.M.mmm(alpha, a.op, b.M, b.op, beta, out)

    def integrate(self, out=None, hermetian=False):
        a = self.a
        b = self.b
        assert a.op == 'C' or a.M.dtype == float and a.op == 'N'
        assert b.op == 'N'
        return a.M.integrate(b.M, out, hermetian)


class AtomBlockMatrix:
    def __init__(self, M_aii):
        self.M_aii = list(M_aii)

    def mmm(self, alpha, opa, b, opb, beta, out):
        assert opa == 'N'
        assert opb == 'N'
        assert beta == 0.0
        I1 = 0
        for M_ii in self.M_aii:
            I2 = I1 + len(M_ii)
            out.a[I1:I2] = np.dot(M_ii, b.a[I1:I2])
            I1 = I2
        return out


class ProjectionMatrix(Matrix):
    def __init__(self, nproj_a, nbands, gd, bcomm, rank_a):
        self.indices = []
        I1 = 0
        for a, ni in enumerate(nproj_a):
            if gd.comm.rank == rank_a[a]:
                I2 = I1 + ni
                self.indices.append((a, I1, I2))
                I1 = I2

        Matrix.__init__(self, I2, nbands, dist=(bcomm, 1, -1))
        self.rank_a = rank_a

    def items(self):
        for a, I1, I2 in self.indices:
            yield a, self.a[I1:I2].T


class HMMM:
    def __init__(self, M, comm, dtype, P_ani, dist):
        if isinstance(P_ani, dict):
            self.atom_indices = []
            self.slices = []
            I1 = 0
            for a, P_ni in P_ani.items():
                I2 = I1 + P_ni.shape[1]
                self.atom_indices.append(a)
                self.slices.append((I1, I2))
                I1 = I2

            P_nI = np.empty((M, I1), dtype)
            for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
                P_nI[:, I1:I2] = P_ni
        else:
            P_nI = P_ani

        SpatialMatrix.__init__(self, M, P_nI.shape[1], dtype, P_nI, dist)
        self.comm = comm

    def new(self):
        P_nI = np.empty_like(self.a)
        pm = ProjectionMatrix(self.shape[0], self.comm, self.dtype, P_nI,
                              self.dist)
        pm.atom_indices = self.atom_indices
        pm.slices = self.slices
        return pm

    def __iter__(self):
        P_nI = self.data
        for I1, I2 in self.slices:
            yield P_nI[:, I1:I2]

    def extract_to(self, P_ani):
        P_nI = self.a
        for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
            P_ni[:] = P_nI[:, I1:I2]

    def mmm(self, alpha, opa, b, opb, beta, c):
        if isinstance(b, PAWMatrix):
            assert (alpha, beta, opa, opb) == (1, 0, 'N', 'N')
            for (I1, I2), M_ii in zip(self.slices, b.M_aii):
                c.a[:, I1:I2] = np.dot(self.a[:, I1:I2], M_ii)
        else:
            SpatialMatrix.mmm(self, alpha, opa, b, opb, beta, c)
