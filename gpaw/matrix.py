import numpy as np

import gpaw.utilities.lapack as lapack
import _gpaw


global_blacs_context_store = {}


class NoDistribution:
    def __init__(self, M, N):
        self.shape = (M, N)
        
    def mmm(self, alpha, a, opa, b, opb, beta, destination):
        # print(self is b, self is b.source)
        c = np.dot(op(opa, a), op(opb, b)).reshape(destination.data.shape)
        if beta == 0:
            destination.data[:] = alpha * c
        else:
            assert beta == 1
            destination.data += alpha * c
        
    def inverse_cholesky(self, data):
        lapack.inverse_cholesky(data)
        
    def diagonalize(self, data, eps_n):
        lapack.diagonalize(data, eps_n)
        
        
class BLACSDistribution:
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
        
    def mmm(self, alpha, a, opa, b, opb, beta, destination):
        M, Ka = a.shape
        Kb, N = b.shape
        if opa == 'T':
            M, Ka = Ka, M
        if opb == 'T':
            Kb, N = N, Kb
        if opa == 'C' and a.dtype == float:
            opa = 'N'
        _gpaw.pblas_gemm(N, M, Ka, alpha, b.data, a.data,
                         beta, destination.data,
                         b.dist.desc, a.dist.desc, destination.dist.desc,
                         opb, opa)

    
def create_distribution(M, N, comm=None, r=1, c=1, b=None):
    if r == c == 1:
        return NoDistribution()
    return BLACSDistribution(M, N, comm, r, c, b)
      

class Matrix:
    def __init__(self, M, N, dtype=None, data=None, dist=None):
        self.shape = (M, N)
        
        if dtype is None:
            if data is None:
                dtype = float
            else:
                dtype = data.dtype
        self.dtype = dtype

        if isinstance(dist, tuple):
            dist = create_distribution(M, N, *dist)
        self.dist = dist

        if data is None:
            self.data = np.empty(dist.shape, self.dtype)
        else:
            self.data = np.asarray(data)

        self.source = None
        
        self.comm_to_be_summed_over = None
        
        self.comm = None

    def __str__(self):
        return str(self.data)

    def new(self):
        return Matrix(*self.shape, dtype=self.dtype, dist=self.dist)

    def finish_sums(self):
        if self.comm_to_be_summed_over:
            self.comm_to_be_summed_over.sum(self.data, 0)
        self.comm_to_be_summed_over = None
        
    def __setitem__(self, i, x):
        # assert i == slice(None)
        x.eval(self)

    def eval(self, destination, beta=0):
        assert destination.dist == self.dist
        if beta == 0:
            destination.data[:] = self.data
        else:
            assert beta == 1
            destination.data += self.data

    def __iadd__(self, x):
        x.eval(self, 1.0)
        return self

    def __mul__(self, x):
        if isinstance(x, Matrix):
            x = ('N', x)
        return Product(('N', self), x)

    def __rmul__(self, x):
        return Product(x, ('N', self))

    def __or__(self, x):
        return Product(self.dv, Product(('C', self), ('T', x)))

    @property
    def T(self):
        return Product(('T', self))

    @property
    def C(self):
        return Product(('C', self))

    def mmm(self, alpha, opa, b, opb, beta, destination):
        self.dist.mmm(alpha, self, opa, b, opb, beta, destination)

    def inverse_cholesky(self):
        self.finish_sums()
        self.dist.inverse_cholesky(self.data)
        
    def diagonalize(self, eps_n):
        self.finish_sums()
        self.dist.diagonalize(self.data, eps_n)

        
class RealSpaceMatrix(Matrix):
    def __init__(self, M, gd, dtype=None, data=None, dist=None):
        N = gd.get_size_of_global_array().prod()
        Matrix.__init__(self, M, N, dtype, data, dist)
        self.data.shape = (-1,) + tuple(gd.n_c)
        self.gd = gd
        self.dv = gd.dv
        self.comm = gd.comm

    def new(self, buf=None):
        return RealSpaceMatrix(self.shape[0], self.gd, self.dtype, buf,
                               self.dist)


class PWExpansionMatrix(Matrix):
    def __init__(self, M, pd, data=None, dist=None):
        Matrix.__init__(self, M, data.shape[1], complex, data, dist)
        self.pd = pd
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def new(self, buf=None):
        if buf is not None:
            buf = buf.ravel()[:self.data.size]
            buf.shape = self.data.shape
        return PWExpansionMatrix(self.shape[0], self.pd, buf, self.dist)

    def mmm(self, alpha, opa, b, opb, beta, destination):
        if (self.pd.dtype == float and opa in 'NC' and
            isinstance(b, PWExpansionMatrix)):
            assert opa == 'C' and opb == 'T' and beta == 0
            a = self.data.view(float)
            b = b.data.view(float)
            destination.data[:] = np.dot(a, b.T)
            destination.data *= 2 * alpha
            destination.data -= alpha * np.outer(a[:, 0], b[:, 0])
        else:
            Matrix.mmm(self, alpha, opa, b, opb, beta, destination)


class ProjectorMatrix(Matrix):
    def __init__(self, M, comm, dtype, P_ani, dist):
        if isinstance(P_ani, dict):
            self.slices = []
            I1 = 0
            for a, P_ni in P_ani.items():
                I2 = I1 + P_ni.shape[1]
                self.slices.append((I1, I2))
                I1 = I2

            P_nI = np.empty((M, I1), dtype)
            for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
                P_nI[:, I1:I2] = P_ni
        else:
            P_nI = P_ani

        Matrix.__init__(self, M, P_nI.shape[1], dtype, P_nI, dist)
        self.comm = comm

    def new(self):
        P_nI = np.empty_like(self.data)
        pm = ProjectorMatrix(self.shape[0], self.comm, self.dtype, P_nI,
                             self.dist)
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


def op(opx, x):
    x = x.data.reshape((len(x.data), -1))
    if opx == 'N':
        return x
    if opx == 'T':
        return x.T
    if opx == 'C':
        return x.conj()
    return x.T.conj()


class Product:
    def __init__(self, x, y=None):
        if y is None:
            self.things = [x]
        elif isinstance(y, Product):
            if isinstance(x, Product):
                self.things = x.things + y.things
            else:
                self.things = [x] + y.things
        else:
            self.things = [x, y]

    def __str__(self):
        return str(self.things)

    def eval(self, destination, beta=0):
        if isinstance(self.things[0], (int, float)):
            alpha = self.things.pop(0)
        else:
            alpha = 1
        a, b = self.things
        self.things = None
        if callable(a):
            opb, b = b
            assert beta == 0
            assert alpha == 1
            assert opb == 'N'
            a(b, destination)
            destination.source = b
        else:
            opa, a = a
            opb, b = b
            a.mmm(alpha, opa, b, opb, beta, destination)
            if opa in 'NC' and a.comm:
                destination.comm_to_be_summed_over = a.comm
                assert opb in 'TH' and b.comm is a.comm

    def __mul__(self, x):
        if isinstance(x, Matrix):
            x = Product(('N', x))
        return Product(self, x)


if __name__ == '__main__':
    from gpaw.mpi import world
    from gpaw.grid_descriptor import GridDescriptor
    gd = GridDescriptor([2, 3, 4], [2, 3, 4])
    N = 2
    a = RealSpaceMatrix(N, gd, float, dist=(world, world.size))
    a.data[:] = 1
    a.data[0, 0, world.rank] = 0
    c = Matrix(N, N, dist=(world, world.size))

    def f(x, y):
        y.data[:] = x.data + 1

    c[:] = (a | a)
    print(c.data)
    c.inverse_cholesky()
    print(c.data)
    b = a.new()
    b[:] = c.C * a
    c[:] = (b | b)
    print(c.data);asdgf
    c[:] = (a | b)
    b[:] = f * a
    c[:] = (a | b)
    d = a.new()
    d[:] = c * a
    print(c)
    c += a * b.T
    print(c)
