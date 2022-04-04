import numpy as np
from gpaw.core.matrix import Matrix


def test_invcholesky():
    S = Matrix(2, 2, data=np.array([[1.0, 117], [0.1j, 2.0]]))
    L = S.copy()
    L.invcholesky()
    S.data[0, 1] = -0.1j
    print(L.data @ S.data @ L.data.T.conj())
    A = L.multiply(S, opa='N').multiply(L, opb='C')
    print(A.data)
    assert abs(A.data - np.eye(2)).max() < 1e-14
