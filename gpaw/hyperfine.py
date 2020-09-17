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
# from gpaw.utilities import unpack2
from gpaw.gaunt import gaunt


def hyper(spin_density_R, gd, spos_ac, ecut=None):
    print(spin_density_R.shape)
    #import matplotlib.pyplot as plt
    #x = np.linspace(-4/0.529177, 5.1/0.529177, 60, 0)
    # x = np.linspace(-4/0.529177, 4/0.529177, 42, 0)
    #print(x)
    #plt.plot(x,
    #         spin_density_R[:, 27, 27] * 0.6666666666666666)
    # plt.plot(np.linspace(-4/0.529177, 4/0.529177, 42, 0),
    #          spin_density_R[:, 21, 21] * 0.6666666666666666)
    pd = PWDescriptor(ecut, gd, complex)

    spin_density_G = pd.fft(spin_density_R)
    #print(spin_density_G[0], gd.cell_cv, spos_ac)
    G_Gv = pd.get_reciprocal_vectors()
    eiGR_aG = np.exp(1j * spos_ac.dot(gd.cell_cv).dot(G_Gv.T))
    W1_a = pd.integrate(spin_density_G, eiGR_aG) / gd.dv * (2 / 3)
    #print(W1_a)
    #spos_ac = np.zeros((60, 3)) + 0.5
    #spos_ac[:, 0] = np.linspace(0, 1, 60, 0)
    # spos_ac[:, 0] = x / (8/0.529177) + .5
    #eiGR_aG = np.exp(1j * spos_ac.dot(gd.cell_cv).dot(G_Gv.T))

    #W1_a = pd.integrate(spin_density_G, eiGR_aG) / gd.dv * (2 / 3)
    #plt.plot(x, W1_a)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    spin_density_G /= G2_G

    W2_vva = np.empty((3, 3, len(spos_ac)))
    for v1 in range(3):
        for v2 in range(3):
            W_a = pd.integrate(G_Gv[:, v1] * G_Gv[:, v2] * spin_density_G,
                               eiGR_aG)
            W2_vva[v1, v2] = -W_a

    W2_a = np.trace(W2_vva) / 3
    for v in range(3):
        W2_vva[v, v] -= W2_a

    return W1_a, W2_vva


def paw_correction(spin_density_ii, setup):
    print(spin_density_ii.diagonal())
    D0_jj = expand(spin_density_ii, setup.l_j, 0)[0]

    phit_jg = np.array(setup.data.phit_jg)
    phi_jg = np.array(setup.data.phi_jg)

    rgd = setup.rgd

    nt0 = phit_jg[:, 0].dot(D0_jj).dot(phit_jg[:, 0]) / (4 * np.pi)**0.5
    n0 = phit_jg[:, 0].dot(D0_jj).dot(phi_jg[:, 0]) / (4 * np.pi)**0.5
    print(n0*0.666, nt0*0.666)
    nt0_g = np.einsum('ab, ag, bg -> g', D0_jj, phit_jg, phit_jg)
    rgd.plot(nt0_g / (4 * np.pi)**0.5 * 0.666, show=1)
    D2_mjj = expand(spin_density_ii, setup.l_j, 2)
    nt2_mg = np.einsum('mab, ag, bg -> mg', D2_mjj, phit_jg, phit_jg)
    w_g = rgd.poisson(nt2_mg[0], 2)
    # rgd.plot(nt2_mg[0])
    # rgd.plot(w_g, show=1)


def expand(D_ii, l_j, l):
    G_LLm = gaunt(lmax=2)[:, :, l**2:(l + 1)**2]
    D_mjj = np.empty((2 * l + 1, len(l_j), len(l_j)))
    i1a = 0
    for j1, l1 in enumerate(l_j):
        i1b = i1a + 2 * l1 + 1
        i2a = 0
        for j2, l2 in enumerate(l_j):
            i2b = i2a + 2 * l2 + 1
            D_mjj[:, j1, j2] = np.einsum('ab, abm -> m',
                                         D_ii[i1a:i1b, i2a:i2b],
                                         G_LLm[l1**2:(l1 + 1)**2,
                                               l2**2:(l2 + 1)**2])
            i2a = i2b
        i1a = i1b
    return D_mjj
