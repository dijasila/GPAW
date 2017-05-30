import numpy as np

from gpaw.matrix.matrix import Matrix, Product
from gpaw.lfc import LFC


class SpatialMatrix(Matrix):
    def __init__(self, M, N, dtype, data, dist):
        Matrix.__init__(self, M, N, dtype, data, dist)
        self.comm_to_be_summed_over = None
        self.comm = None

    def finish_sums(self):
        if self.comm_to_be_summed_over and not self.transposed:
            self.comm_to_be_summed_over.sum(self.a, 0)
        self.comm_to_be_summed_over = None

    def mmm(self, alpha, opa, b, opb, beta, c):
        Matrix.mmm(self, alpha, opa, b, opb, beta, c)
        if opa in 'NC' and self.comm:
            c.comm_to_be_summed_over = self.comm
            assert opb in 'TH' and b.comm is self.comm

    def integrate(self, other, out, hermitian=False):
        if out is None:
            out = Matrix(len(self), len(other))
        self.mmm(self.dv, 'C', other, 'T', 0.0, out)
        return out

    def apply(self, func, out):
        func(self.array, out.array)

    def __setitem__(self, i, x):
        if isinstance(i, int):
            self.array[i] = x;asdf
        else:
            Matrix.__setitem__(self, i, x)

    def __or__(self, other):
        return Product(self.dv, (self, 'C'), (other, 'T'))

    def __getitem_______________(self, i):
        assert self.dist.shape[0] == self.shape[0]
        return self.array[i]


class UniformGridFunctions(SpatialMatrix):
    def __init__(self, M, gd, dtype=None, data=None, kpt=None, dist=None,
                 collinear=True):
        N = gd.get_size_of_global_array().prod()
        SpatialMatrix.__init__(self, M, N, dtype, data, dist)
        if data is None:
            self.a = self.a.reshape(-1).reshape(self.dist.shape)
            self.transposed = False
            print(self.a.flags)
        self.array = self.a.reshape((-1,) + tuple(gd.n_c))
        self.gd = gd
        self.dv = gd.dv
        self.comm = gd.comm

    def __repr__(self):
        s = SpatialMatrix.__repr__(self).split('(')[1][:-1]
        shape = self.gd.get_size_of_global_array()
        s = 'UniformGridMatrix({}, gpts={}x{}x{})'.format(s, *shape)
        return s

    def new(self, buf=None):
        return UniformGridFunctions(self.shape[0], self.gd, self.dtype, buf,
                                    None, self.dist)

    def plot(self):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111)
        a, b, c = self.array.shape[1:]
        ax.plot(self.array[0, a // 2, b // 2])
        plt.show()


class PlaneWaveExpansions(SpatialMatrix):
    def __init__(self, M, pd, data, q=-1, dist=None):
        orig = data
        if pd.dtype == float:
            data = data.view(float)
        SpatialMatrix.__init__(self, M, data.shape[1], pd.dtype, data, dist)
        self.array = orig
        self.pd = pd
        self.q = q
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def __getitem______________________(self, i):
        assert self.distribution.shape[0] == self.shape[0]
        return self.array[i]

    def new(self, buf=None):
        if buf is not None:
            buf = buf.ravel()[:self.array.size]
            buf.shape = self.array.shape
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


class AtomCenteredFunctions:
    dtype = float

    def __init__(self, desc, functions):
        self.lfc = LFC(desc, functions)
        self.atom_indices = []
        self.slices = []
        I1 = 0
        for a, f in enumerate(functions):
            I2 = I1 + len(f)
            self.atom_indices.append(a)
            self.slices.append((I1, I2))
            I1 = I2
        self.nfuncs = I2

    def __len__(self):
        return self.nfuncs

    @property
    def C(self):
        return self

    def set_positions(self, spos):
        print(spos)
        self.lfc.set_positions(spos)

    positions = property(None, set_positions)

    def eval(self, out):
        out.array[:] = 0.0
        coef_M = np.zeros(len(self))
        for M, a in enumerate(out.array):
            coef_M[M] = 1.0
            self.lfc.lfc.lcao_to_grid(coef_M, a, -1)
            coef_M[M] = 0.0

    def integrate(self, other, out, hermetian):
        self.lfc.integrate(other.array, self.dictview(out), -1)

    def dictview(self, matrix):
        M_In = matrix.array
        M_ani = {}
        for a, (I1, I2) in zip(self.atom_indices, self.slices):
            M_ani[a] = M_In[I1:I2].T
        return M_ani


class AtomBlockMatrix:
    def __init__(self, I1, I2, M_aii):
        self.M_aii = M_aii

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


class UniformGridDensity:
    pass


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

    def mmm(self, alpha, opa, b, opb, beta, c):
        if isinstance(b, PAWMatrix):
            assert (alpha, beta, opa, opb) == (1, 0, 'N', 'N')
            for (I1, I2), M_ii in zip(self.slices, b.M_aii):
                c.a[:, I1:I2] = np.dot(self.a[:, I1:I2], M_ii)
        else:
            SpatialMatrix.mmm(self, alpha, opa, b, opb, beta, c)
