import numpy as np
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.spline import Spline
a = 4.0
domain = Domain(cell=[a, a + 1, a + 2], pbc=(0, 1, 1))
gd = GridDescriptor(domain, N_c=[16, 20, 20])
spos_ac = np.array([[0.25, 0.15, 0.35], [0.5, 0.5, 0.5]])
kpts_kc = None
s = Spline(l=0, rmax=2.0, f_g=np.array([1, 0.9, 0.1, 0.0]))
p = Spline(l=1, rmax=2.0, f_g=np.array([1, 0.9, 0.1, 0.0]))
spline_aj = [[s], [s, p]]
c = LFC(gd, spline_aj, cut=True, forces=True)
if kpts_kc is not None:
    c.set_k_points(kpts_kc)
c.set_positions(spos_ac)
C_ani = c.dict(3, zero=True)
if 1 in C_ani:
    C_ani[1][:, 1:] = np.eye(3)
psi = gd.zeros(3)
c.add(psi, C_ani)
c.integrate(psi, C_ani)
if 1 in C_ani:
    d = C_ani[1][:, 1:].diagonal()
    assert d.ptp() < 4e-6
    C_ani[1][:, 1:] -= np.diag(d)
    assert abs(C_ani[1]).max() < 8e-18
d_aniv = c.dict(3, derivative=True)
c.derivative(psi, d_aniv)
pos_av = np.dot(spos_ac, domain.cell_cv)
if 1 in d_aniv:
    for v in range(3):
        assert abs(d_aniv[1][v - 1, 0, v] + 0.2144) < 5e-5
        d_aniv[1][v - 1, 0, v] = 0
    assert abs(d_aniv[1]).max() < 2e-17
eps = 0.0001
for v in range(3):
    pos_av[0, v] += eps
    c.set_positions(np.dot(pos_av, domain.icell_cv.T))
    c.integrate(psi, C_ani)
    C0_n = C_ani[0][:, 0].copy()
    pos_av[0, v] -= 2 * eps
    c.set_positions(np.dot(pos_av, domain.icell_cv.T))
    c.integrate(psi, C_ani)
    C0_n -= C_ani[0][:, 0]
    C0_n /= -2 * eps
    assert abs(C0_n - d_aniv[0][:, 0, v]).max() < 7e-6
 

    
