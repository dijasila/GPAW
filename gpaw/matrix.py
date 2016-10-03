import numpy as np


class RealSpaceGridDescriptor:
    def __init__(self, gd):
        self.dv = gd.dv
        self.sum = gd.comm.sum
        self.size = gd.n_c.prod()

    def convert_to_raw_data(self, data):
        return data


class ProjectorDescriptor:
    def __init__(self, P_ani):
        self.slices = []
        I1 = 0
        for a, P_ni in P_ani.items():
            I2 = I1 + P_ni.shape[1]
            self.slices.append((I1, I2))
            I1 = I2
        self.I = I1

    def sum(self, x):
        pass

    def convert_to_raw_data(self, P_ani):
        if self.I == 0:
            return np.zeros((0, 0))

        for P_ni in P_ani.values():
            P_nI = np.empty((P_ni.shape[0], self.I), P_ni.dtype)
            break
        for P_ni, (I1, I2) in zip(P_ani.values(), self.slices):
            P_nI[:, I1:I2] = P_ni
        return P_nI

    def extract(self, P_nI, P_ani):
        I1 = 0
        for a, P_ni in P_ani.items():
            I2 = I1 + P_ni.shape[1]
            P_ni[:] = P_nI[:, I1:I2]
            I1 = I2


class SimpleDescriptor:
    def convert_to_raw_data(self, data):
        return data

    def sum(self, x):
        pass


def op(opx, x):
    x = x.data.reshape((len(x.data), -1))
    if opx == 'N':
        return x
    if opx == 'T':
        return x.T
    if opx == 'C':
        return x.conj()
    return x.T.conj()


class Matrix:
    def __init__(self, data, descriptor=None, raw=False):
        if raw:
            self.data = data
        else:
            if isinstance(data, dict):
                descriptor = ProjectorDescriptor(data)
            elif descriptor:
                descriptor = RealSpaceGridDescriptor(descriptor)
            else:
                data = np.asarray(data)
                descriptor = SimpleDescriptor()

            self.data = descriptor.convert_to_raw_data(data)

        self.descriptor = descriptor
        self.source = None

    def __str__(self):
        return str(self.data)

    def empty_like(self):
        return Matrix(np.empty_like(self.data), self.descriptor, raw=True)

    def __iter__(self):
        for I1, I2 in self.descriptor.slices:
            yield self.data[:, I1:I2]

    def extract(self, target):
        self.descriptor.extract(self.data, target)

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
        return Product(self.descriptor.dv, Product(('C', self), ('T', x)))

    @property
    def T(self):
        return Product(('T', self))

    @property
    def C(self):
        return Product(('C', self))

    @property
    def H(self):
        return Product(('H', self))


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
                a.descriptor.sum(c)
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
    a = Matrix(p1, gd)
    b = Matrix(p2, gd)
    c = Matrix(np.zeros((2, 2)))

    def f(x):
        return x + 1

    c[:] = (a | a)
    c[:] = (a | b)
    b[:] = f * a
    c[:] = (a | b)
    b[:] = c * a
    print(c)
    c += a * b.T
    print(c)
