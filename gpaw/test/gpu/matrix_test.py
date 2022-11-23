from gpaw.gpu import cupy as cp
from gpaw.core.matrix import Matrix
import numpy as np


def test_syrk():
    a = np.array([[1, 1, 1], [1, 0.5, -1]])
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    print(b.data)
    a = cp.asarray(a)
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    print(b.data, b.xp)


def test_zyrk():
    a = np.array([[1, 1 + 2j, 1], [1, 0.5j, -1 - 0.5j]])
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    print(b.data)
    a = cp.asarray(a)
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    print(b.data, b.xp)
    print(a)
    c = cp.cublas.syrk('N', a, lower=True)
    print(c)
    print(a @ a.T.conj())
