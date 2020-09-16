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


def hyper(spin_density_R, gd, spos_ac, ecut=None):
    pd = PWDescriptor(ecut, gd)

    spin_density_G = pd.fft(spin_density_R)

    G_Gv = pd.get_reciprocal_vectors()
    eiGR_Ga = np.exp(2j * np.pi * G_Gv.dot(gd.cell_cv.T.dot(spos_ac.T)))

    W1_a = spin_density_G.dot(eiGR_Ga) * (2 / 3)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    G_Gv /= G2_G[:, np.newaxis]**0.5
    W2_vva = np.einsum('Gi, Gi, G, Ga -> ija',
                       G_Gv, G_Gv, spin_density_G, eiGR_Ga)
    W2_vva -= spin_density_G.dot(eiGR_Ga) / 3

    return W1_a - W2_vva


def paw_correction(D_sp, setup):
    s = setup
    print(dir(setup))
    print(dir(setup.data))
    print(s.l_j, s.n_j, np.pi**.5)
    setup.rgd.plot(s.data.phi_jg[0])
    setup.rgd.plot(s.data.phit_jg[0], show=True)
    print((s.data.phi_jg[0]**2 * s.rgd.r_g**2).dot(s.rgd.dr_g))
    