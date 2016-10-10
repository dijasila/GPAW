import numpy as np

from gpaw.matrix.matrix import Matrix, Product


class SpatialMatrix(Matrix):
    def __init__(self, M, N, dtype, data, dist):
        Matrix.__init__(self, M, N, dtype, data, dist)
        self.comm_to_be_summed_over = None
        self.comm = None

    def finish_sums(self):
        if self.comm_to_be_summed_over and not self.transposed:
            self.comm_to_be_summed_over.sum(self.x, 0)
        self.comm_to_be_summed_over = None

    def mmm(self, alpha, opa, b, opb, beta, c):
        Matrix.mmm(self, alpha, opa, b, opb, beta, c)
        if opa in 'NC' and self.comm:
            c.comm_to_be_summed_over = self.comm
            assert opb in 'TH' and b.comm is self.comm

    def matrix_elements(self, other, M, hermitian=False):
        if self is other:
            pass
        self.mmm(self.dv, 'C', other, 'T', 0.0, M)

    def project(self, lfc, P_nI):
        lfc.integrate(self.A, P_nI.dictview(), self.q)

    def apply(self, func, out):
        func(self.A, out.A)

    def __setitem__(self, i, x):
        if isinstance(i, int):
            self.A[i] = x
        else:
            Matrix.__setitem__(self, i, x)

    def __or__(self, other):
        return Product(self.dv, (self, 'C'), (other, 'T'))

    def __getitem__(self, i):
        assert self.dist.shape[0] == self.shape[0]
        return self.A[i]


class UniformGridMatrix(SpatialMatrix):
    def __init__(self, M, gd, dtype=None, data=None, q=-1, dist=None):
        N = gd.get_size_of_global_array().prod()
        SpatialMatrix.__init__(self, M, N, dtype, data, dist)
        if data is None:
            self.a = self.a.T
            self.transposed = False
        self.A = self.a.reshape((-1,) + tuple(gd.n_c))
        self.gd = gd
        self.dv = gd.dv
        self.comm = gd.comm
        self.q = q

    def new(self, buf=None):
        return UniformGridMatrix(self.shape[0], self.gd, self.dtype, buf,
                                 self.q, self.dist)


class PWExpansionMatrix(SpatialMatrix):
    def __init__(self, M, pd, data, q=-1, dist=None):
        self.A = data
        if pd.dtype == float:
            data = data.view(float)
        SpatialMatrix.__init__(self, M, data.shape[1], pd.dtype, data, dist)
        self.pd = pd
        self.q = q
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def __getitem__(self, i):
        assert self.distribution.shape[0] == self.shape[0]
        return self.A[i]

    def new(self, buf=None):
        if buf is not None:
            buf = buf.ravel()[:self.A.size]
            buf.shape = self.A.shape
        return PWExpansionMatrix(self.shape[0], self.pd, buf, self.q,
                                 self.dist)

    def mmm(self, alpha, opa, b, opb, beta, c):
        if (self.pd.dtype == float and opa in 'NC' and
            isinstance(b, PWExpansionMatrix)):
            assert opa == 'C' and opb == 'T' and beta == 0
            a = self
            c.a[:] = np.dot(a.a, b.a.T)
            c.a *= 2 * alpha
            c.a -= alpha * np.outer(a.a[:, 0], b.a[:, 0])
        else:
            SpatialMatrix.mmm(self, alpha, opa, b, opb, beta, c)


class ProjectionMatrix(SpatialMatrix):
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

    def dictview(self):
        P_nI = self.a
        P_ani = {}
        for a, (I1, I2) in zip(self.atom_indices, self.slices):
            P_ani[a] = P_nI[:, I1:I2]
        return P_ani

    def mmm(self, alpha, opa, b, opb, beta, c):
        if isinstance(b, PAWMatrix):
            assert (alpha, beta, opa, opb) == (1, 0, 'N', 'N')
            for (I1, I2), M_ii in zip(self.slices, b.M_aii):
                c.a[:, I1:I2] = np.dot(self.a[:, I1:I2], M_ii)
        else:
            SpatialMatrix.mmm(self, alpha, opa, b, opb, beta, c)


class PAWMatrix:
    def __init__(self, M_aii):
        self.M_aii = list(M_aii)
