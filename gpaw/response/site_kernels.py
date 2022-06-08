"""Compute site-kernels. Used for computing Heisenberg exchange.
Specifically, one maps DFT calculations onto a Heisenberg lattice model,
where the site-kernels define the lattice sites and magnetic moments."""

import numpy as np
from scipy.special import jv
from gpaw.response.susceptibility import get_pw_coordinates
from ase.units import Bohr


def sinc(x):
    """np.sinc(x) = sin(pi*x) / (pi*x), hence the division by pi"""
    return np.sinc(x / np.pi)


def site_kernel_interface(pd, sitePos_mv, shapes_m='sphere',
                          rc_m=1.0, zc_m='diameter'):
    """Compute site kernels using an arbitrary combination of shapes for the
        integration region at different magnetic sites.
        Accepts spheres, cylinders and unit cell.

    Parameters
    ----------
    pd : Planewave Descriptor. Contains mixed information about
            plane-wave basis
    sitePos_mv : Array with positions of magnetic sites within unit cell
    shapes_m : List of str or str. Which shapes to use for the
            different integration regions.
            Options are 'sphere', 'cylinder', 'unit cell'.
            If 'unit cell' then rc_rm, zc_rm, sitePos_mv do nothing and
                a single site in the middle of the unit cell is used.
    rc_m : Radius of integration region
    zc_m : Height of integration cylinders (if any)
            If 'diameter' then zc=2*rc
            If 'unit cell' then use height of unit cell (makes sense in 2D)
    """

    # Number of sites
    if shapes_m == 'unit cell':
        nsites = 1
    else:
        nsites = len(sitePos_mv)

    # Reformat shape parameters
    if type(shapes_m) is str:
        shapes_m = np.array([shapes_m] * nsites)
    if type(rc_m) in [int, float]:
        rc_m = np.array([rc_m] * nsites)
    if type(zc_m) in [int, float]:
        zc_m = np.array([zc_m] * nsites)
    if type(zc_m) == str:
        zc_m = [zc_m] * nsites
    for m in range(nsites):
        # Reformat zc if needed
        if zc_m[m] == 'unit cell':
            zc_m[m] = np.sum(pd.gd.cell_cv[:, -1]) * Bohr   # Units of Å.
        elif zc_m[m] == 'diameter':
            zc_m[m] = 2. * rc_m[m]
    zc_m = np.array(zc_m, dtype=float)
    ez_v = np.array([0., 0., 1.])  # Should be made an input in the future XXX

    # Convert input units (Å) to atomic units (Bohr)
    sitePos_mv = sitePos_mv / Bohr
    rc_m = rc_m / Bohr
    zc_m = zc_m / Bohr

    # Construct Fourier components
    G_Gv, q_v, Omega_cell = _extract_pd_info(pd)
    Q_GGv = _construct_wave_vectors(G_Gv, q_v)

    # Array to fill
    K_GGm = np.zeros(Q_GGv.shape[:2] + (nsites,), dtype=np.complex128)

    # --- The Calculation itself --- #

    # Loop through magnetic sites
    # Should be vectorized? XXX
    for m in range(nsites):
        # Get site specific values
        shape, rc, zc = shapes_m[m], rc_m[m], zc_m[m]
        sitePos_v = sitePos_mv[m, :]

        # Compute complex prefactor
        prefactor = _makePrefactor(sitePos_v, Q_GGv, Omega_cell)

        # Do computation for relevant shape
        if shape == 'sphere':
            K_GG = spherical_geometry_factor(Q_GGv, rc)

        elif shape == 'cylinder':
            K_GG = cylindrical_geometry_factor(Q_GGv, ez_v, rc, zc)

        elif shape == 'unit cell':
            # Get real-space basis vectors
            # Give the user control over these XXX
            a1, a2, a3 = pd.gd.cell_cv

            # # Default site position is center of unit cell
            # # This should not be up to some secret functionality to decide XXX
            # if sitePos_v is None:
            #     sitePos_v = 1 / 2 * (a1 + a2 + a3)
            K_GG = K_unit_cell(Q_GGv, a1, a2, a3)

        else:
            print('Not a recognised shape')

        # Update data
        K_GGm[:, :, m] = prefactor * K_GG

    return K_GGm


def spherical_geometry_factor(Q_Qv, rc):
    """Calculate the site centered geometry factor for a spherical site kernel:

           /
    Θ(Q) = | dr e^(-iQ.r) θ(|r|<r_c)
           /

           4πr_c
         = ‾‾‾‾‾ [sinc(|Q|r_c) - cos(|Q|r_c)]
           |Q|^2

                    3 [sinc(|Q|r_c) - cos(|Q|r_c)]
         = V_sphere ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                              (|Q|r_c)^2

    where the dimensionless geometry factor satisfies:

    Θ(Q)/V_sphere --> 1 for |Q|r_c --> 0.

    Parameters
    ----------
    Q_Qv : np.ndarray
        Wave vectors to evaluate the site centered geometry factor at. The
        cartesian coordinates needs to be the last dimension of the array (v),
        but the preceeding index/indices Q can have any tensor structure, such
        that Q_Qv.shape = (..., 3).
    rc : float
        Radius of the sphere.
    """
    assert Q_Qv.shape[-1] == 3
    assert isinstance(rc, float) and rc > 0.

    # Calculate the sphere volume
    Vsphere = 4 * np.pi * rc**3. / 3

    # Calculate |Q|r_c
    Qrc_Q = np.linalg.norm(Q_Qv, axis=-1) * rc

    # Allocate array with ones to provide the correct dimensionless geometry
    # factor in the |Q|r_c --> 0 limit.
    # This is done to avoid division by zero.
    Theta_Q = np.ones(Q_Qv.shape[:-1], dtype=float)

    # Calculate the dimensionless geometry factor
    Qrcs = Qrc_Q[Qrc_Q > 1.e-8]
    Theta_Q[Qrc_Q > 1.e-8] = 3. * (sinc(Qrcs) - np.cos(Qrcs)) / Qrcs**2.

    Theta_Q *= Vsphere

    return Theta_Q


def cylindrical_geometry_factor(Q_Qv, ez_v, rc, hc):
    """Calculate the site centered geometry factor for a cylindrical site kernel:

           /
    Θ(Q) = | dr e^(-iQ.r) θ(ρ<r_c) θ(|z|/2<h_c)
           /

            4πr_c
         = ‾‾‾‾‾‾‾ J_1(Q_ρ r_c) sin(Q_z h_c / 2)
           Q_ρ Q_z

                      2 J_1(Q_ρ r_c)
         = V_cylinder ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ sinc(Q_z h_c / 2)
                         Q_ρ r_c

    where z denotes the cylindrical axis, ρ the radial axis and the
    dimensionless geometry factor satisfy:

    Θ(Q)/V_cylinder --> 1 for Q --> 0.

    Parameters
    ----------
    Q_Qv : np.ndarray
        Wave vectors to evaluate the site centered geometry factor at. The
        cartesian coordinates needs to be the last dimension of the array (v),
        but the preceeding index/indices Q can have any tensor structure, such
        that Q_Qv.shape = (..., 3).
    ez_v : np.ndarray
        Normalized direction of the cylindrical axis.
    rc : float
        Radius of the cylinder.
    hc : float
        Height of the cylinder.
    """
    assert Q_Qv.shape[-1] == 3
    assert ez_v.shape == (3,)
    assert abs(np.linalg.norm(ez_v) - 1.) < 1.e-8
    assert isinstance(rc, float) and rc > 0.
    assert isinstance(hc, float) and hc > 0.

    # To do: Make it possible to input the cylindrical axis XXX

    # Calculate cylinder volume
    Vcylinder = np.pi * rc**2. * hc

    # Calculate Q_z h_c and Q_ρ r_c
    Qzhchalf_Q = np.abs(Q_Qv @ ez_v) * hc / 2.
    Qrhorc_Q = np.linalg.norm(np.cross(Q_Qv, ez_v), axis=-1) * rc

    # Allocate array with ones to provide the correct dimensionless geometry
    # factor in the Q_ρ r_c --> 0 limit.
    # This is done to avoid division by zero.
    Theta_Q = np.ones(Q_Qv.shape[:-1], dtype=float)

    # Calculate the dimensionless geometry factor
    Qrhorcs = Qrhorc_Q[Qrhorc_Q > 1.e-8]
    Theta_Q[Qrhorc_Q > 1.e-8] = 2. * jv(1, Qrhorcs) / Qrhorcs
    Theta_Q *= Vcylinder * sinc(Qzhchalf_Q)

    return Theta_Q


def K_unit_cell(Q_GGv, a1, a2, a3):
    """Compute site-kernel for a spherical integration region"""

    # Calculate the parallelepiped volume
    cell_cv = np.array([a1, a2, a3])
    Vparlp = abs(np.linalg.det(cell_cv))

    # Calculate the site-kernel
    K_GG = Vparlp * sinc(Q_GGv @ a1 / 2) * sinc(Q_GGv @ a2 / 2) * \
        sinc(Q_GGv @ a3 / 2)

    return K_GG


def _makePrefactor(sitePos_v, sum_GGv, Omega_cell):
    """Make the complex prefactor which occurs for all site-kernels,
    irrespective of shape of integration region"""
    # Phase factor
    phaseFactor_GG = np.exp(-1.j * sum_GGv @ sitePos_v)

    # Scale factor
    scaleFactor = 1. / Omega_cell

    return scaleFactor * phaseFactor_GG


def _extract_pd_info(pd):
    """Get relevant quantities from pd object (plane-wave descriptor)
    In particular reciprocal space vectors and unit cell volume
    Note : all in Bohr and absolute coordinates."""
    q_qc = pd.kd.bzk_kc
    assert len(q_qc) == 1
    q_c = q_qc[0, :]  # Assume single q
    G_Gc = get_pw_coordinates(pd)

    # Convert to cartesian coordinates
    B_cv = 2.0 * np.pi * pd.gd.icell_cv  # Coordinate transform matrix
    q_v = np.dot(q_c, B_cv)  # Unit = Bohr^(-1)
    G_Gv = np.dot(G_Gc, B_cv)

    # Get unit cell volume in bohr^3
    Omega_cell = pd.gd.volume

    return G_Gv, q_v, Omega_cell


def _construct_wave_vectors(G_Gv, q_v):
    """Construct wave vector array with shape (NG, NG, 3)"""
    NG = len(G_Gv)
    G1_GGv = np.tile(G_Gv[:, np.newaxis, :], [1, NG, 1])
    G2_GGv = np.tile(G_Gv[np.newaxis, :, :], [NG, 1, 1])
    q_GGv = np.tile(q_v[np.newaxis, np.newaxis, :], [NG, NG, 1])

    Q_GGv = G1_GGv - G2_GGv + q_GGv  # G_1 - G_2 + q

    return Q_GGv
