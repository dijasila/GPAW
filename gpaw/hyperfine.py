"""Hyperfine parameters.

See:

    First-principles calculations of defects in oxygen-deficient
    silica exposed to hydrogen

    Peter E. Blöchl

    Phys. Rev. B 62, 6158 – Published 1 September 2000

    https://doi.org/10.1103/PhysRevB.62.6158

"""
import numpy as np

from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.utilities import unpack2


def hyper(spin_density_R, gd, spos_ac, ecut=None):
    pd = PWDescriptor(ecut, gd)#, complex)

    spin_density_G = pd.fft(spin_density_R)

    G_Gv = pd.get_reciprocal_vectors()
    print(spin_density_R.shape)
    n = spin_density_R.shape[0] // 2
    print(spin_density_R[n, n, n - 1:n + 2])
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(-4/0.529177, 4/0.529177, 2 * n, False),
             spin_density_R[n,n])
    # print(spin_density_R.sum() * gd.dv)
    eiGR_Ga = np.exp(-1j * G_Gv.dot(gd.cell_cv.T.dot(spos_ac.T)))

    print(pd.integrate(spin_density_G, eiGR_Ga[:, 0]) / gd.dv)

    W1_a = spin_density_G.dot(eiGR_Ga) * (2 / 3)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    G_Gv /= G2_G[:, np.newaxis]**0.5
    W2_vva = np.einsum('Gi, Gj, G, Ga -> ija',
                       G_Gv, G_Gv, spin_density_G, eiGR_Ga)
    W2_vva -= spin_density_G.dot(eiGR_Ga) / 3

    # print(W1_a, W2_vva)

    return W1_a - W2_vva


def paw_correction(spin_density_ii, setup):
    print(D_sp[0])
    D_ii = unpack2(D_sp[0])
    s = setup
    print(s.l_j, s.n_j)
    # setup.rgd.plot(s.data.phi_jg[0])
    phit_jg = s.data.phi_jg[:2]
    n = np.einsum('ig, jg, ij -> g',
                  phit_jg, phit_jg, D_ii[:2, :2]) / (4 * np.pi)
    print((s.data.phi_jg[0]**2 * s.rgd.r_g**2).dot(s.rgd.dr_g))
    print(n[0])
    s.rgd.plot(n, show=True)
    