from itertools import combinations_with_replacement, product

import numpy as np

from matplotlib import pyplot as plt
#from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gpaw.response.integrators import TetrahedronIntegrator
from gpaw.response.chi0 import ArrayDescriptor


def unit(x_c):
    return np.array([[1]], complex)


def unit_sphere(x_c):
    return np.array([1.0 - (x_c**2).sum()**0.5])

# Test tetrahedron integrator
integrator = TetrahedronIntegrator()
omega_w = np.linspace(0, 3, 50)
gi_w = np.zeros(len(omega_w), float)
I_kw = np.zeros((4, len(omega_w)), float)
de_k = np.array([0, 1, 2, 3], float)
for iw, omega in enumerate(omega_w):
    if de_k[0] <= omega <= de_k[1]:
        case = 0
    elif de_k[1] <= omega <= de_k[2]:
        case = 1
    elif de_k[2] <= omega <= de_k[3]:
        case = 2

    gi, I_k = integrator.get_kpoint_weight(omega, de_k, case)

    gi_w[iw] = gi
    I_kw[:, iw] = I_k

plt.plot(omega_w, gi_w, label='g')
plt.plot(omega_w, I_kw[0], label='I0')
plt.plot(omega_w, I_kw[1], label='I1')
plt.plot(omega_w, I_kw[2], label='I2')
plt.plot(omega_w, I_kw[3], label='I3')
plt.legend()
plt.show()

wd = ArrayDescriptor([0.001])
vol = 1
deps_kM = np.array([[1], [0], [2], [3]], float)
I_KMw = integrator.calculate_integration_weights(wd, deps_kM, vol)

# Calculate surface area of unit sphere
x_g = np.linspace(-1, 1, 30)
x_gc = np.array([comb for comb in product(*([x_g] * 3))])

td = integrator.tesselate(x_gc)
vol = 0
for S in range(td.nsimplex):
    vol += integrator.get_simplex_volume(td, S)

print('Integration volume is: {0}.'.format(vol))

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for simplex in td.simplices:
        points_pv = td.points[simplex]
        for (i, p1), (j, p2) in product(enumerate(points_pv),
                                        enumerate(points_pv)):
            if i <= j:
                continue
            p_pv = np.array([p1, p2])
            ax.plot(p_pv[:, 0], p_pv[:, 1], p_pv[:, 2], 'k-')

    plt.show()

domain = (td,)
out_wxx = np.zeros((1, 1, 1), complex)
integrator.integrate('response_function', domain,
                     (unit, unit_sphere),
                     ArrayDescriptor([0.]),
                     out_wxx=out_wxx)

print(out_wxx)
