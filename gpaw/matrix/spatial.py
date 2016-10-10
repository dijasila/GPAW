import numpy as np

from gpaw.matrix import Matrix


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


class UniformGridMatrix(SpatialMatrix):
    def __init__(self, M, gd, dtype=None, data=None, dist=None):
        N = gd.get_size_of_global_array().prod()
        SpatialMatrix.__init__(self, M, N, dtype, data, dist)
        if data is None:
            self.data = self.data.T
            self.transposed = False
        self.x = self.data
        self.data = self.x.reshape((-1,) + tuple(gd.n_c))
        self.gd = gd
        self.dv = gd.dv
        self.comm = gd.comm

    def __getitem__(self, i):
        assert self.distribution.shape[0] == self.shape[0]
        return self.data[i]

    def new(self, buf=None):
        return UniformGridMatrix(self.shape[0], self.gd, self.dtype, buf,
                                 self.dist)

    def project(self, lfc, P_nI):
        lfc.integrate(self.data, P_nI.dictview())
        return P_nI


class PWExpansionMatrix(SpatialMatrix):
    def __init__(self, M, pd, data=None, k=-1, dist=None):
        if pd.dtype == float:
            orig = data
            data = data.view(float)
        SpatialMatrix.__init__(self, M, data.shape[1], pd.dtype, data, dist)
        if pd.dtype == float:
            self.data = orig
        self.pd = pd
        self.k = k
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def __getitem__(self, i):
        assert self.distribution.shape[0] == self.shape[0]
        return self.data[i]

    def new(self, buf=None):
        if buf is not None:
            buf = buf.ravel()[:self.data.size]
            buf.shape = self.data.shape
        return PWExpansionMatrix(self.shape[0], self.pd, buf, self.dist)

    def mmm(self, alpha, opa, b, opb, beta, c):
        if (self.pd.dtype == float and opa in 'NC' and
            isinstance(b, PWExpansionMatrix)):
            assert opa == 'C' and opb == 'T' and beta == 0
            a = self.data.view(float)
            b = b.data.view(float)
            c.data[:] = np.dot(a, b.T)
            c.data *= 2 * alpha
            c.data -= alpha * np.outer(a[:, 0], b[:, 0])
        else:
            SpatialMatrix.mmm(self, alpha, opa, b, opb, beta, c)

    def project(self, lfc, P_nI):
        lfc.integrate(self.data, P_nI.dictview(), self.k)
        return P_nI


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
        P_nI = np.empty_like(self.data)
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
        P_nI = self.data
        for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
            P_ni[:] = P_nI[:, I1:I2]

    def dictview(self):
        P_nI = self.data
        P_ani = {}
        for a, (I1, I2) in zip(self.atom_indices, self.slices):
            P_ani[a] = P_nI[:, I1:I2]
        return P_ani

    def mmm(self, alpha, opa, b, opb, beta, c):
        if isinstance(b, PAWMatrix):
            assert (alpha, beta, opa, opb) == (1, 0, 'N', 'N')
            for (I1, I2), M_ii in zip(self.slices, b.M_aii):
                c.x[:, I1:I2] = np.dot(self.x[:, I1:I2], M_ii)
        else:
            SpatialMatrix.mmm(self, alpha, opa, b, opb, beta, c)


class PAWMatrix:
    def __init__(self, M_aii):
        self.M_aii = list(M_aii)
