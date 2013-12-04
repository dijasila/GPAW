"""Wigner-Seitz truncated coulomb interaction.

See:

    Ravishankar Sundararaman and T. A. Arias:
    Phys. Rev. B 87, 165122 (2013)
    
    Regularization of the Coulomb singularity in exact exchange by
    Wigner-Seitz truncated interactions: Towards chemical accuracy
    in nontrivial systems
"""

from math import pi

import numpy as np

from gpaw.utilities import erf
from gpaw.grid_descriptor import GridDescriptor


def wigner_seitz_truncated_coulomb(pd, nk_c):
    cell_cv = pd.gd.cell_cv * nk_c[:, np.newaxis]
    gd = GridDescriptor(pd.gd.N_c, cell_cv)
    rc = 0.5 * (gd.icell_cv**2).sum(1).max()**-0.5
    a = 5 / rc
    v_R = gd.empty()
    v_i = v_R.ravel()
    pos_iv = gd.get_grid_point_coordinates().reshape((3, -1)).T
    corner_jv = np.dot(np.indices((2, 2, 2)).reshape((3, 8)).T, cell_cv)
    for i, pos_v in enumerate(pos_iv):
        r = ((pos_v - corner_jv)**2).sum(axis=1).min()**0.5
        if r == 0:
            v_i[i] = 2 * a / pi**0.5
        else:
            v_i[i] = erf(a * r) / r
    K_Q = np.fft.fftn(v_R) * gd.dv
    q_c = pd.kd.bzk_kc[0]
    shift_c = abs((q_c * nk_c).round().astype(int))
    K_Q = K_Q[shift_c[0]::nk_c[0],
              shift_c[1]::nk_c[1],
              shift_c[2]::nk_c[2]]
    assert not (gd.N_c % 2).any()
    max_c = gd.N_c // nk_c
    K_G = pd.zeros()
    for G, Q in enumerate(pd.Q_qG[0]):
        Q_c = (np.unravel_index(Q, gd.N_c) +
               gd.N_c // 2) % gd.N_c - gd.N_c // 2
        if (abs(Q_c) < max_c).all():
            K_G[G] = K_Q[tuple(Q_c)]
    K_G[0] += pi / a**2
    G2_G = pd.G2_qG[0][1:]
    K_G[1:] += 4 * pi * (1 - np.exp(-G2_G / (4 * a**2))) / G2_G
    return K_G

    
if __name__ == '__main__':
    from gpaw.kpt_descriptor import KPointDescriptor
    from gpaw.wavefunctions.pw import PWDescriptor
    gd = GridDescriptor([16, 16, 16], [2, 2, 2])
    kd = KPointDescriptor([[0, 0, 0]])
    pd = PWDescriptor(5.0, gd, complex, kd=kd)
    wigner_seitz_truncated_coulomb(pd, np.array([2, 2, 2]))
