import numpy as np

from gpaw.gaunt import gaunt
from gpaw.sphere.integrate import radial_truncation_function


def calculate_site_pair_density_correction(pawdata, rcut_p, drcut, lambd_p):
    r"""Calculate PAW correction to the site pair density.

    For each pair of partial waves, the PAW correction to the site pair density
    is given by:

     ap     0,mi,mi' /                  a    a     ̰ a   ̰ a
    N    = g         | r^2 dr θ(r<rc) [φ(r) φ(r) - φ(r) φ(r)]
     ii'    0,li,li' /         p        i    i'     i    i'

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
    np = len(rcut_p)
    assert len(lambd_p) == np
    theta_pg = [radial_truncation_function(rgd.r_g, rcut, drcut, lambd)
                for rcut, lambd in zip(rcut_p, lambd_p)]
    N_pii = np.zeros((np, ni, ni), dtype=float)

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
                    gaunt_coeff = G_LLL[0, l1**2 + m1, l2**2 + m2]
                    if gaunt_coeff == 0:
                        continue
                    # Set up the i=(l,m) index for each partial wave
                    i1 = i1_counter + m1
                    i2 = i2_counter + m2

                    # Integrate correction
                    for p, theta_g in enumerate(theta_pg):
                        N_pii[p, i1, i2] += gaunt_coeff * rgd.integrate_trapz(
                            theta_g * dn_g)

            # Add to i and i' counters
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return N_pii
