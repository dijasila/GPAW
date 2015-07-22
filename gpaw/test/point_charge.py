from __future__ import division
import numpy as np
from gpaw.test import equal
import _gpaw

c = np.linalg.solve([[1, 1, 1, 1],
                     [0, 2, 4, 6],
                     [0, 2, 12, 30],
                     [1 / 3, 1 / 5, 1 / 7, 1 / 9]],
                    [1, -1, 2, 0.5])
print(c)
print(c * 32)
x = np.linspace(0, 1, 101)
v = np.polyval(c[::-1], x**2)
equal((v * x**2 - x).sum() / 100, 0, 1e-5)

if 0:
    import matplotlib.pyplot as plt
    plt.plot(x, v)
    x = np.linspace(0.2, 1.5, 101)
    plt.plot(x, 1 / x)
    plt.show()

h = 0.1
q = 2.2
beg_v = np.zeros(3, int)
h_v = np.ones(3) * h
q_p = np.ones(1) * q
R_pv = np.array([[0.1, 0.2, -0.3]])
vext_G = np.zeros((1, 1, 1))
rhot_G = np.ones((1, 1, 1))


def f(rc):
    vext_G[:] = 0.0
    _gpaw.pc_potential(beg_v, h_v, q_p, R_pv, rc, vext_G)
    return vext_G[0, 0, 0]

d = (R_pv[0]**2).sum()**0.5

for rc in [0.9 * d, 1.1 * d]:
    if d > rc:
        v0 = q / d
    else:
        v0 = np.polyval(c[::-1], (d / rc)**2) * q / rc

    equal(f(rc), v0, 1e-12)
    
    F_pv = np.zeros((1, 3))
    _gpaw.pc_potential(beg_v, h_v, q_p, R_pv, rc, vext_G, rhot_G, F_pv)
    eps = 0.0001
    for v in range(3):
        R_pv[0, v] += eps
        ep = f(rc)
        R_pv[0, v] -= 2 * eps
        em = f(rc)
        R_pv[0, v] += eps
        F = -(ep - em) / (2 * eps) * h**3
        equal(F, F_pv[0, v], 1e-9)
