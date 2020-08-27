import numpy as np
from gpaw.tetrahedron import TetrahedronMethod

rcell = np.diag([1, 0.1, 0.1])


def f(N):
    k = (np.linspace(-0.5, 0.5, N, endpoint=False) + 0.5 / N) * 2 * np.pi
    e = -np.cos(k)[:, np.newaxis]
    f = np.empty((N, 1))
    w = np.zeros(N) + 1 / N
    t = TetrahedronMethod(rcell, (N, 1, 1))
    t._calculate(0.5, e, w, f)
    return f.sum() / N, (f * e).sum() / N * np.pi


for N in range(4, 100, 8):
    print(f(N))
