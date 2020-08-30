import numpy as np
from gpaw.tetrahedron import TetrahedronMethod
from gpaw.dos import DOSCalculator

rcell = np.diag([1, 0.1, 0.1])


def f(N):
    k = (np.linspace(-0.5, 0.5, N, endpoint=False) + 0.5 / N) * 2 * np.pi
    e = -np.cos(k)[:, np.newaxis]
    f = np.empty((N, 1))
    w = np.zeros(N) + 1 / N
    t = TetrahedronMethod(rcell, (N, 1, 1))
    ef, _ = t._calculate(0.5, e, w, f)
    wfs = WFS(e, ef)
    dos = DOSCalculator(wfs, ef, ef, 1, cell=np.linalg.inv(rcell))
    df = dos.dos(spin=0, width=0.0).get_weights()[0]
    return f.sum() / N, (f * e).sum() / N * np.pi + 1, ef, df * np.pi


class WFS:
    def __init__(self, e, ef):
        self.eig_skn = e[np.newaxis]
        self.fermi_level = ef
        self.size = (len(e), 1, 1)
        self.bz2ibz_map = None

    def weights(self):
        n = self.size[0]
        return np.zeros(n) + 1 / n

    def eigenvalues(self):
        return self.eig_skn


for N in range(4, 100, 8):
    print(f(N))
