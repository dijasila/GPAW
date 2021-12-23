"""Compute site-kernels. Used for computing Heisenberg exchange.
Specifically, one maps DFT calculations onto a Heisenberg lattice model,
where the site-kernels define the lattice sites and magnetic moments."""

import numpy as np


def sinc(x):
    """np.sinc(x) = sin(pi*x) / (pi*x), hence the division by pi"""
    return np.sinc(x / np.pi)


def calc_K_unit_cell(pd, sitePos_v=None):
    """
    Compute site-kernel for a spherical integration region

    :param pd: Planewave Descriptor. Contains mixed information about
               planewave basis.
    :param sitePos_v: nd.array
                    Position of site in unit cell

    :return: NG X NG matrix
    """

    # Get reciprocal lattice vectors and q-vector from pd
    q_qc = pd.kd.bzk_kc
    assert len(q_qc) == 1
    q_c = q_qc[0, :]     # Assume single q
    G_Gc = get_pw_coordinates(pd)
    NG = len(G_Gc)

    # Convert to cartesian coordinates
    B_cv = 2.0 * np.pi * pd.gd.icell_cv  # Coordinate transform matrix
    q_v = np.dot(q_c, B_cv)  # Unit = Bohr^(-1)
    G_Gv = np.dot(G_Gc, B_cv)

    # Get unit cell vectors and volume in Bohr
    Omega_cell = pd.gd.volume
    a1, a2, a3 = pd.gd.cell_cv

    # Default is center of unit cell
    if sitePos_v is None:
        sitePos_v = 1/2 * (a1 + a2 + a3)

    # Construct arrays
    G1_GGv = np.tile(G_Gv[:, np.newaxis, :], [1, NG, 1])
    G2_GGv = np.tile(G_Gv[np.newaxis, :, :], [NG, 1, 1])
    q_GGv = np.tile(q_v[np.newaxis, np.newaxis, :], [NG, NG, 1])

    # Combine arrays
    sum_GGv = G1_GGv + G2_GGv + q_GGv  # G_1 + G_2 + q

    # Compute site-kernel
    K_GG = sinc(sum_GGv@a1 / 2) * sinc(sum_GGv@a2 / 2) * sinc(sum_GGv@a3 / 2)

    # Multiply by prefactor
    # e^{i (G_1 + G_2 + q) . (a_1 + a_2 + a_3)/2}
    phase_factor_GG = np.exp(1j * sum_GGv @ sitePos_v)
    K_GG = K_GG * np.sqrt(2/Omega_cell) * phase_factor_GG

    return K_GG
