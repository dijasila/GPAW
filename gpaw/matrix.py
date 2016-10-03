import numpy as np


def matrix(data, descriptor=None):
    if isinstance(data, dict):
        return ProjectorMatrix(data, descriptor)
    if descriptor:
        if data.ndim == 2:
            return PWExpansionMatrix(data, descriptor.pd)
        else:
            return RealSpaceMatrix(data, descriptor.gd)
    return Matrix(data)


class Matrix:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.source = None

    def __str__(self):
        return str(self.data)

    def sum(self, x):
        pass

    def empty_like(self):
        return Matrix(np.empty_like(self.data))

    def __setitem__(self, i, x):
        #assert i == slice(None)
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


class RealSpaceMatrix(Matrix):
    def __init__(self, data, gd):
        Matrix.__init__(self, data)
        self.gd = gd
        self.dv = gd.dv
        self.sum = gd.comm.sum

    def empty_like(self):
        return RealSpaceMatrix(np.empty_like(self.data), self.gd)


class PWExpansionMatrix(Matrix):
    def __init__(self, data, pd):
        Matrix.__init__(self, data)
        self.pd = pd
        self.dv = pd.gd.dv / pd.gd.N_c.prod()

    def empty_like(self):
        return PWExpansionMatrix(np.empty_like(self.data), self.pd)


class ProjectorMatrix(Matrix):
    def __init__(self, P_ani=None, comm=None):
        self.comm = comm
        if P_ani is not None:
            self.slices = []
            I1 = 0
            N = 0
            dtype = float
            for a, P_ni in P_ani.items():
                N, i = P_ni.shape
                dtype = P_ni.dtype
                I2 = I1 + i
                self.slices.append((I1, I2))
                I1 = I2

            P_nI = np.empty((N, I1), dtype)
            for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
                P_nI[:, I1:I2] = P_ni

            Matrix.__init__(self, P_nI)

    def empty_like(self):
        pm = ProjectorMatrix(comm=self.comm)
        pm.data = np.empty_like(self.data)
        pm.slices = self.slices
        return pm

    def sum(self, x):
        self.comm.sum(x)

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
            # print(a is b, a is b.source)
            c = np.dot(op(opa, a), op(opb, b)).reshape(destination.data.shape)
            if opa in 'NC':
                a.sum(c)
            if beta == 0:
                destination.data[:] = alpha * c
            else:
                assert beta == 1
                destination.data += alpha * c

    def __mul__(self, x):
        return Product(self, x)


if __name__ == '__main__':
    from gpaw.grid_descriptor import GridDescriptor
    gd = GridDescriptor([2, 3, 4], [2, 3, 4])
    p1 = gd.zeros(2)
    p2 = gd.zeros(2)
    p1[:] = 1
    p2[:] = 2
    a = matrix(p1, gd)
    b = matrix(p2, gd)
    c = matrix(np.zeros((2, 2)))

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
