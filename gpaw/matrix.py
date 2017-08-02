import numpy as np

import scipy.linalg as linalg
import gpaw.utilities.blas as blas
import _gpaw
from gpaw.utilities import pack2, unpack


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
        return ('BLACSDistribution(global={0}, local={1}, blocksize={2})'
                .format(*('{0}x{1}'.format(*shape)
                          for shape in [self.desc[2:4:1],
                                        self.shape,
                                        self.desc[4:6:1]])))

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
            self.array = np.empty(dist.shape, self.dtype, order='F')
        else:
            self.array = data.reshape(dist.shape)

        self.transposed = self.array.flags['F_CONTIGUOUS']

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
            1 / 0  # sssssself.array[:] = x
        else:
            x.eval(self)

    def __array_____(self, dtype):
        assert self.dtype == dtype
        assert self.dist.serial
        return self.array

    def evallllllllll(self, destination=None, beta=0):
        if destination is None:
            destination = 42
        assert destination.dist == self.dist
        if beta == 0:
            destination.array[:] = self.array
        else:
            assert beta == 1
            destination.array += self.array
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

    def cholesky(self):
        self.finish_sums()
        self.dist.cholesky(self.array)

    def inv(self):
        self.finish_sums()
        self.dist.inv(self.array)

    def eigh(self, eps_n):
        self.finish_sums()
        self.dist.eigh(self.array, eps_n)


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


class AtomBlockMatrix:
    def __init__(self, M_asii, nspins=None, comm=None, size_a=None):
        self.M_asii = M_asii
        self.nspins = nspins
        self.comm = comm
        self.size_a = size_a

        self.rank_a = None

    def multiply(self, alpha, opa, P1_In, opb, beta, P2_In):
        assert opa == 'N'
        assert opb == 'N'
        assert beta == 0.0

        for a, I1, I2 in P2_In.indices:
            M_ii = self.M_asii[a]
            if M_ii.ndim == 3:
                M_ii = M_ii[P2_In.spin]
            P2_In.array[I1:I2] = np.dot(M_ii, P1_In.array[I1:I2])

        return P2_In

    def broadcast(self):
        M_asii = []
        for a, ni in enumerate(self.size_a):
            M_sii = self.M_asii.get(a)
            if M_sii is None:
                M_sii = np.empty((self.nspins, ni, ni))
            self.comm.broadcast(M_sii, self.rank_a[a])
            M_asii.append(M_sii)
        return M_asii

    def pack(self):
        M_asii = self.broadcast()
        P = sum(ni * (ni + 1) // 2 for ni in self.size_a)
        M_sP = np.empty((self.nspins, P))
        P1 = 0
        for ni, M_sii in zip(self.size_a, M_asii):
            P2 = P1 + ni * (ni + 1) // 2
            M_sP[:, P1:P2] = [pack2(M_ii) for M_ii in M_sii]
            P1 = P2
        return M_sP

    def unpack(self, M_sP):
        assert len(self.M_asii) == 0
        if M_sP is None:
            return
        P1 = 0
        for a, ni in enumerate(self.size_a):
            P2 = P1 + ni * (ni + 1) // 2
            self.M_asii[a][:] = [unpack(M_p) for M_p in M_sP[:, P1:P2]]
            P1 = P2


class ProjectionMatrix(Matrix):
    def __init__(self, nproj_a, nbands, acomm, bcomm, rank_a,
                 collinear=True, spin=0, dtype=float):
        self.nproj_a = nproj_a
        self.acomm = acomm
        self.bcomm = bcomm
        self.rank_a = rank_a
        self.collinear = collinear
        self.spin = spin

        self.indices = []
        self.my_atom_indices = []
        I1 = 0
        for a, ni in enumerate(nproj_a):
            if acomm.rank == rank_a[a]:
                self.my_atom_indices.append(a)
                I2 = I1 + ni
                self.indices.append((a, I1, I2))
                I1 = I2

        Matrix.__init__(self, I1, nbands, dtype, dist=(bcomm, 1, -1))

    def new(self):
        return ProjectionMatrix(
            self.nproj_a, self.shape[1], self.acomm, self.bcomm,
            self.rank_a, self.collinear, self.spin, self.dtype)

    def add_product(self, factor, dS_II, P_In, eps_n):
        assert factor == -1.0
        for a, I1, I2 in P_In.indices:
            dS_ii = dS_II.M_asii[a]
            self.array[I1:I2] -= np.dot(dS_ii, P_In.array[I1:I2] * eps_n)

    def items(self):
        for a, I1, I2 in self.indices:
            yield a, self.array[I1:I2].T

    def todict(self):
        return dict(self.items())


"""
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
        P_nI = np.empty_like(self.array)
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
        P_nI = self.array
        for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
            P_ni[:] = P_nI[:, I1:I2]

    def multiply(self, alpha, opa, b, opb, beta, c):
        if isinstance(b, PAWMatrix):
            assert (alpha, beta, opa, opb) == (1, 0, 'N', 'N')
            for (I1, I2), M_ii in zip(self.slices, b.M_aii):
                c.array[:, I1:I2] = np.dot(self.array[:, I1:I2], M_ii)
        else:
            SpatialMatrix.multiply(self, alpha, opa, b, opb, beta, c)
"""
