"""Compute site-kernels. Used for computing Heisenberg exchange.
Specifically, one maps DFT calculations onto a Heisenberg lattice model,
where the site-kernels define the lattice sites and magnetic moments."""

import numpy as np
from scipy.special import jv
from gpaw.response.susceptibility import get_pw_coordinates
from ase.units import Bohr


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

    # Change name of sitePos_mv to positions XXX

    # Number of sites
    if shapes_m == 'unit cell':
        nsites = 1
        # Overwrite positions
        # This is an extremely bad behaviour! Change this XXX
        sitePos_mv = np.array([np.sum(pd.gd.cell_cv, axis=0) / 2.])
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

    # Set up geometries manually for now to be backwards compatible XXX
    geometries = []
    for shape, rc, zc in zip(shapes_m, rc_m, zc_m):
        if shape == 'sphere':
            geometries.append((shape, (rc,)))
        elif shape == 'cylinder':
            # Cylindrical axis should be made an input in the future XXX
            ez_v = np.array([0., 0., 1.])
            hc = zc  # Should be made more reasonable in the future XXX
            geometries.append((shape, (ez_v, rc, hc)))
        elif shape == 'unit cell':  # Change to parallelepiped XXX
            # Give the user control over the cell in the future XXX
            cell_cv = pd.gd.cell_cv  # unit cell of atomic structure
            geometries.append(('parallelepiped', (cell_cv,)))
        else:
            raise ValueError('Invalid site kernel shape:', shape)

    return calculate_site_kernels(pd, sitePos_mv, geometries)


def calculate_site_kernels(pd, positions, geometries):
    """Calculate the sublattice site kernel:

                1  /
    K_aGG'(q) = ‾‾ | dr e^(-i[G-G'+q].r) θ(r∊V_a)
                V0 /

    where V_a denotes the integration volume of site a, centered at the site
    position τ_a, and V0 denotes the unit cell volume.

    In the calculation, the kernel is split in two contributions:

    1) The Fourier component of the site position:

    τ_a(Q) = e^(-iQ.τ_a)

    2) The site centered geometry factor

           /
    Θ(Q) = | dr e^(-iQ.r) θ(r+τ_a∊V_a)
           /

    where Θ(Q) only depends on the geometry of the integration volume.
    With this:

                1
    K_aGG'(q) = ‾‾ τ_a(G-G'+q) Θ(G-G'+q)
                V0

    Parameters
    ----------
    pd : PWDescriptor
        Plane wave descriptor corresponding to the q wave vector of interest.
    positions : np.ndarray
        Site positions. Array of shape (nsites, 3).
    geometries : list
        List of site geometries. A site geometry is a tuple of the integration
        volume shape (str) and arguments (tuple): (shape, args). Valid shapes
        are 'sphere', 'cylinder' and 'parallelepiped'. The integration volume
        arguments specify the size and orientation of the integration region.

    Returns
    -------
    K_aGG : np.ndarray (dtype=complex)
        Site kernels of site a and plane wave component G and G'.
    """
    assert positions.shape[0] == len(geometries)
    assert positions.shape[1] == 3

    # Extract unit cell volume
    V0 = pd.gd.volume

    # Construct Q=G-G'+q
    Q_GGv = construct_wave_vectors(pd)

    # Allocate site kernel array
    nsites = len(geometries)
    K_GGa = np.zeros(Q_GGv.shape[:2] + (nsites,), dtype=complex)

    # Calculate the site kernel for each site individually
    for a, (tau_v, (shape, args)) in enumerate(zip(positions, geometries)):

        # Compute the site centered geometry factor
        _geometry_factor = create_geometry_factor(shape)  # factory pattern
        Theta_GG = _geometry_factor(Q_GGv, *args)

        # Compute the Fourier component of the site position
        tau_GG = np.exp(-1.j * Q_GGv @ tau_v)

        # Update data
        K_GGa[:, :, a] = 1 / V0 * tau_GG * Theta_GG

    return K_GGa


def construct_wave_vectors(pd):
    """Construct wave vectors Q=G1-G2+q corresponding to the q-vector of
    interest."""
    G_Gv, q_v = get_plane_waves_and_reduced_wave_vector(pd)

    # Allocate arrays for G, G' and q respectively
    nG = len(G_Gv)
    G1_GGv = np.tile(G_Gv[:, np.newaxis, :], [1, nG, 1])
    G2_GGv = np.tile(G_Gv[np.newaxis, :, :], [nG, 1, 1])
    q_GGv = np.tile(q_v[np.newaxis, np.newaxis, :], [nG, nG, 1])

    # Contruct the wave vector G1 - G2 + q
    Q_GGv = G1_GGv - G2_GGv + q_GGv

    return Q_GGv


def get_plane_waves_and_reduced_wave_vector(pd):
    """Get the reciprocal lattice vectors and reduced wave vector of the plane
    wave representation corresponding to the q-vector of interest."""
    # Get the reduced wave vector
    q_qc = pd.kd.bzk_kc
    assert len(q_qc) == 1
    q_c = q_qc[0, :]  # Assume single q

    # Get the reciprocal lattice vectors in relative coordinates
    G_Gc = get_pw_coordinates(pd)

    # Convert to cartesian coordinates
    B_cv = 2.0 * np.pi * pd.gd.icell_cv  # Coordinate transform matrix
    q_v = np.dot(q_c, B_cv)  # Unit = Bohr^(-1)
    G_Gv = np.dot(G_Gc, B_cv)

    return G_Gv, q_v


def create_geometry_factor(shape):
    """Creator compoenent of geometry factor factory pattern."""
    if shape == 'sphere':
        return spherical_geometry_factor
    elif shape == 'cylinder':
        return cylindrical_geometry_factor
    elif shape == 'parallelepiped':
        return parallelepipedic_geometry_factor

    raise ValueError('Invalid site kernel shape:', shape)


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


def parallelepipedic_geometry_factor(Q_Qv, cell_cv):
    """Calculate the site centered geometry factor for a parallelepipedic site
    kernel:

           /
    Θ(Q) = | dr e^(-iQ.r) θ(r∊V_parallelepiped)
           /

         = |det[a1, a2, a3]| sinc(Q.a1 / 2) sinc(Q.a2 / 2) sinc(Q.a3 / 2)

         = V_parallelepiped sinc(Q.a1 / 2) sinc(Q.a2 / 2) sinc(Q.a3 / 2)

    where a1, a2 and a3 denotes the parallelepipedic cell vectors.

    Parameters
    ----------
    Q_Qv : np.ndarray
        Wave vectors to evaluate the site centered geometry factor at. The
        cartesian coordinates needs to be the last dimension of the array (v),
        but the preceeding index/indices Q can have any tensor structure, such
        that Q_Qv.shape = (..., 3).
    cell_cv : np.ndarray, shape=(3, 3)
        Cell vectors of the parallelepiped, where v denotes the cartesian
        coordinates.
    """
    assert Q_Qv.shape[-1] == 3
    assert cell_cv.shape == (3, 3)

    # Calculate the parallelepiped volume
    Vparlp = abs(np.linalg.det(cell_cv))
    assert Vparlp > 1.e-8  # Not a valid parallelepiped if volume vanishes

    # Calculate the site-kernel
    a1, a2, a3 = cell_cv
    Theta_Q = Vparlp * sinc(Q_Qv @ a1 / 2) * sinc(Q_Qv @ a2 / 2) * \
        sinc(Q_Qv @ a3 / 2)

    return Theta_Q


def sinc(x):
    """np.sinc(x) = sin(pi*x) / (pi*x), hence the division by pi"""
    return np.sinc(x / np.pi)
