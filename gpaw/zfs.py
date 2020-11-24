"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""


def zfs(calc) -> Array2D:
    smooth_part()


class Densities:
    def __init__(self):
        spin
        P_ani -> Q
        psit_G -> psit_R -> n_R -> n_G + Q * g_G
def smooth_part(s1, s2, psit_,
                spos_ac: Array2D) -> Array2D:
    """Contribution from pseudo spin-density."""
    D_vv = np.zeros((3, 3))

    for s1, psit1 in enumerate(psit_s):
        for s2, psit2 in enumerate(psit_s):
            density
    pd = PWDescriptor(ecut, gd)
    spin_density_G = pd.fft(spin_density_R)
    G_Gv = pd.get_reciprocal_vectors()
    eiGR_aG = np.exp(-1j * spos_ac.dot(gd.cell_cv).dot(G_Gv.T))

    # Isotropic term:
    W1_a = pd.integrate(spin_density_G, eiGR_aG) / gd.dv * (2 / 3)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    spin_density_G /= G2_G

    # Anisotropic term:
    W_vva = np.empty((3, 3, len(spos_ac)))
    for v1 in range(3):
        for v2 in range(3):
            W_a = pd.integrate(G_Gv[:, v1] * G_Gv[:, v2] * spin_density_G,
                               eiGR_aG)
            W_vva[v1, v2] = -W_a / gd.dv

    W_a = np.trace(W_vva) / 3
    for v in range(3):
        W_vva[v, v] -= W_a
        W_vva[v, v] += W1_a

    return W_vva.transpose((2, 0, 1))
