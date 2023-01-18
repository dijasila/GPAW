import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp


@pytest.mark.gpu
@pytest.mark.serial
def test_zyrk():
    a = np.array([[1, 1 + 2j, 2], [1, 0.5j, -1 - 0.5j]])
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b.tril2full()
    a = cp.asarray(a)
    m = Matrix(2, 3, data=a)
    b2 = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b2.tril2full()
    c = b2.to_cpu()
    assert (c.data == b.data).all()


@pytest.mark.gpu
@pytest.mark.serial
def test_eigh():
    H = Matrix(2, 2, data=np.array([[2, 42.1 + 42.1j], [0.1 - 0.1j, 3]]))
    S = Matrix(2, 2, data=np.array([[1, 42.1 + 42.2j], [0.1 - 0.2j, 0.9]]))
    h = Matrix(2, 2, data=cp.asarray(H.data))
    s = Matrix(2, 2, data=cp.asarray(S.data))
    E = H.eigh(S)
    e = h.eigh(s)
    assert e == pytest.approx(E)
    assert h.to_cpu().data == pytest.approx(H.data)
