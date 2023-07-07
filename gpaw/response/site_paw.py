import numpy as np

from gpaw.gaunt import gaunt
from gpaw.sphere.rshe import calculate_reduced_rshe
from gpaw.sphere.integrate import radial_truncation_function


def calculate_site_pair_density_correction(pawdata, rcut_p, drcut, lambd_p):
    """Calculate PAW correction to the site pair density.

    Some documentation here! XXX
    """
    # Make a real spherical harmonics expansion of the identity operator
    Y_nL = pawdata.xc_correction.Y_nL
    leb_quad_size = Y_nL.shape[0]
    identity_ng = pawdata.rgd.empty(leb_quad_size)
    identity_ng[:] = 1.
    rshe, _ = calculate_reduced_rshe(identity_ng, Y_nL, lmax=0)

    return calculate_local_site_correction(
        pawdata, rshe, rcut_p, drcut, lambd_p)


def calculate_local_site_correction(pawdata, rshe, rcut_p, drcut, lambd_p):
    r"""

    Update the documentation here! XXX

    For each pair of partial waves, the PAW correction to the site pair density
    is given by:

     ap     __  0,mi,mi' /                  a    a     ̰ a   ̰ a
    N    = √4π g         | r^2 dr θ(r<rc) [φ(r) φ(r) - φ(r) φ(r)]
     ii'        0,li,li' /         p        i    i'     i    i'

    where g refer to the Gaunt coefficients.

    Here, we evaluate the correction N_ii'^ap for various smooth truncation
    functions θ_p(r<rc), parametrized by rc, Δrc and λ.
    """
    rgd = pawdata.rgd
    ni = pawdata.ni  # Number of partial waves
    l_j = pawdata.l_j  # l-index for each radial function index j
    G_LLL = gaunt(max(l_j))
    # (Real) radial functions for the partial waves
    phi_jg = pawdata.data.phi_jg
    phit_jg = pawdata.data.phit_jg

    # Calculate smooth truncation functions and allocate array
    Np = len(rcut_p)
    assert len(lambd_p) == Np
    theta_pg = [radial_truncation_function(rgd.r_g, rcut, drcut, lambd)
                for rcut, lambd in zip(rcut_p, lambd_p)]
    N_pii = np.zeros((Np, ni, ni), dtype=float)

    # Loop of radial function indices for partial waves i and i'
    i1_counter = 0
    for j1, l1 in enumerate(l_j):
        i2_counter = 0
        for j2, l2 in enumerate(l_j):
            # Calculate the radial partial wave correction
            dn_g = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

            # Generate m-indices for each radial function
            for m1 in range(2 * l1 + 1):
                for m2 in range(2 * l2 + 1):
                    # Set up the i=(l,m) index for each partial wave
                    i1 = i1_counter + m1
                    i2 = i2_counter + m2

                    # Loop through the real spherical harmonics of the local
                    # function f(r)
                    for L, f_g in zip(rshe.L_M, rshe.f_gM.T):
                        # Angular integral
                        gaunt_coeff = G_LLL[L, l1**2 + m1, l2**2 + m2]
                        if gaunt_coeff == 0:
                            continue
                        # Radial integral
                        for p, theta_g in enumerate(theta_pg):
                            N_pii[p, i1, i2] += \
                                gaunt_coeff * rgd.integrate_trapz(
                                theta_g * f_g * dn_g)

            # Add to i and i' counters
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return N_pii
