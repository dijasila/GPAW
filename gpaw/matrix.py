import numpy as np

import ase
from gpaw.grid_descriptor import GridDescriptor
import _gpaw


def matrix(data, *args, **kwargs):
    if isinstance(data, dict):
        return ProjectorMatrix(data, *args, **kwargs)
    if args:
        if data.ndim == 2:
            return PWExpansionMatrix(data, args[0].pd, **kwargs)
        else:
            if isinstance(args[0], GridDescriptor):
                return RealSpaceMatrix(data, args[0], **kwargs)
            return RealSpaceMatrix(data, args[0].gd, **kwargs)
    return Matrix(data, **kwargs)


def create_layout(m, n, comm=None, r=1, c=1, b=None):
    # if r == c == 1:
    if comm is None:
        return None
    context = _gpaw.new_blacs_context(comm.get_c_object(), c, r, 'R')
    if b is None:
        br = (m + r - 1) // r
        bc = n
    else:
        br = bc = b
    N, M = _gpaw.get_blacs_local_shape(context, n, m, bc, br, 0, 0)
    lld = max(1, N)
    return np.array([1, context, n, m, bc, br, 0, 0, lld], np.intc)


class Matrix:
    def __init__(self, data, layout=()):
        self.data = np.asarray(data)

        n = len(self.data)
        m = self.data.size // n
        self.layout = create_layout(n, m, *layout)

        self.source = None
        self.summcomm = None
        self.comm = None

    def __str__(self):
        return str(self.data)

    def empty_like(self):
        return Matrix(np.empty_like(self.data))

    def touch(self):
        if self.commsum:
            self.commsum.sum(self.data)

    def __setitem__(self, i, x):
        # assert i == slice(None)
        x.eval(self)

    def eval(self, destination, beta=0):
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

    @property
    def H(self):
        return Product(('H', self))

    def mmm(self, alpha, opa, b, opb, beta, destination):
        if self.layout is not None:
            print(self.layout, alpha, self.data.shape, opa, b.data.shape, opb,
                  beta, destination.data.shape);asdg
            #self.layout.mmm(alpha, self, opa, b, opb, beta, destination)
            return

        # print(self is b, self is b.source)
        c = np.dot(op(opa, self), op(opb, b)).reshape(destination.data.shape)
        if beta == 0:
            destination.data[:] = alpha * c
        else:
            assert beta == 1
            destination.data += alpha * c


class RealSpaceMatrix(Matrix):
    def __init__(self, data, gd, layout):
        Matrix.__init__(self, data, layout)
        self.gd = gd
        self.dv = gd.dv
        self.comm = gd.comm

    def empty_like(self):
        return RealSpaceMatrix(np.empty_like(self.data), self.gd, self.layout)


class PWExpansionMatrix(Matrix):
    def __init__(self, data, pd, layout):
        Matrix.__init__(self, data, layout)
        self.pd = pd
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def empty_like(self):
        return PWExpansionMatrix(np.empty_like(self.data), self.pd)

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
    def __init__(self, P_ani, comm, N=0, dtype=float, layout=()):
        self.comm = comm
        if P_ani is not None:
            self.slices = []
            I1 = 0
            for a, P_ni in P_ani.items():
                N, i = P_ni.shape
                dtype = P_ni.dtype
                I2 = I1 + i
                self.slices.append((I1, I2))
                I1 = I2

            P_nI = np.empty((N, I1), dtype)
            for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
                P_nI[:, I1:I2] = P_ni

            Matrix.__init__(self, P_nI, layout)

    def empty_like(self):
        pm = ProjectorMatrix(None, self.comm)
        pm.data = np.empty_like(self.data)
        pm.slices = self.slices
        pm.layout = self.layout
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
                destination.sumcomm = a.comm
                assert opb in 'TH' and b.comm is a.comm

    def __mul__(self, x):
        if isinstance(x, Matrix):
            x = Product(('N', x))
        return Product(self, x)


if __name__ == '__main__':
    from gpaw.mpi import world
    gd = GridDescriptor([2, 3, 4], [2, 3, 4])
    p1 = gd.zeros(2)
    p2 = gd.zeros(2)
    p1[:] = 1
    p2[:] = 2
    L = (world, 1, 1)
    a = matrix(p1, gd, layout=L)
    b = matrix(p2, gd, layout=L)
    c = matrix(np.zeros((2, 2)), layout=L)

    def f(x, y):
        y.data[:] = x.data + 1

    c[:] = (a | a)
    c[:] = (a | b)
    b[:] = f * a
    c[:] = (a | b)
    d = a.empty_like()
    d[:] = c * a
    print(c)
    c += a * b.T
    print(c)
