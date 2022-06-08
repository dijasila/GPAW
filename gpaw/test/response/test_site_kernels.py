"""Test the site kernel calculation functionality of the response code"""
import numpy as np
import scipy.special as sc

from ase.build import bulk

from gpaw import GPAW, PW
from gpaw.response.site_kernels import (site_kernel_interface,
                                        spherical_geometry_factor,
                                        cylindrical_geometry_factor)
from gpaw.response.susceptibility import get_pw_coordinates


# ---------- Main test ---------- #


def test_site_kernels():
    spherical_kernel_test()
    cylindrical_kernel_test()
    parallelepipedic_kernel_test()
    Co_hcp_test()


# ---------- Actual tests ---------- #


def spherical_kernel_test():
    """Check the numerics of the spherical kernel"""
    # ---------- Inputs ---------- #

    # Relative wave vector lengths to check (relative to 1/rc)
    Qrel_Q = np.array([0.,
                       np.pi / 2,
                       np.pi])

    # Expected results (assuming |Q| * rc = 1.)
    Vsphere = 4. * np.pi / 3.
    test_K_Q = Vsphere * np.array([1.,
                                   3 / (np.pi / 2)**3.,
                                   3 / (np.pi)**2.])

    # Spherical radii to check
    nr = 5
    rc_r = np.random.rand(nr)

    # Wave vector directions to check
    nd = 41
    Q_dv = 2. * np.random.rand(nd, 3) - 1.
    Q_dv /= np.linalg.norm(Q_dv, axis=1)[:, np.newaxis]  # normalize

    # ---------- Script ---------- #

    # Set up wave vectors
    Q_Qdv = Qrel_Q[:, np.newaxis, np.newaxis] * Q_dv[np.newaxis, ...]

    for rc in rc_r:
        # Calculate site centered geometry factor with rescaled wave vector
        K_Qd = spherical_geometry_factor(Q_Qdv / rc, rc)
        # Check against expected result
        assert np.allclose(K_Qd, rc**3. * test_K_Q[:, np.newaxis])


def cylindrical_kernel_test():
    """Check the numerics of the spherical kernel"""
    # ---------- Inputs ---------- #

    # Relative wave vectors (relative to 1/rc) in radial direction
    Qrhorel_Q1 = np.array([0.] + list(sc.jn_zeros(1, 4)))  # Roots of J1(x)

    # Relative wave vectors (relative to 2/hc) in cylindrical direction
    Qzrel_Q2 = list(np.pi * np.arange(5))  # Roots of sin(x)
    Qzrel_Q2 += list(np.pi * np.arange(4) + np.pi / 2)  # Extrema of sin(x)
    Qzrel_Q2 = np.array(Qzrel_Q2)

    # Expected results for roots of J1 (assuming rc=1. and hc=2.)
    Vcylinder = 2. * np.pi
    nQ2 = 11  # Choose random Q_z r_z
    test_Krho_Q1 = np.array([1., 0., 0., 0., 0.])
    Qzrand_Q2 = 10. * np.random.rand(nQ2)
    sinc_zrand_Q2 = np.sin(Qzrand_Q2) / Qzrand_Q2
    test_Krho_Q1Q2 = Vcylinder * test_Krho_Q1[:, np.newaxis]\
        * sinc_zrand_Q2[np.newaxis, :]

    # Expected results for roots and extrema of sin (assuming rc=1. and hc=2.)
    nQ1 = 15  # Choose random Q_ρ h_c
    test_Kz_Q2 = [1., 0., 0., 0., 0.]  # Nodes in sinc(Q_z h_c)
    test_Kz_Q2 += list(np.array([1., -1., 1., -1.]) / Qzrel_Q2[5:])  # Extrema
    test_Kz_Q2 = np.array(test_Kz_Q2)
    Qrhorand_Q1 = 10. * np.random.rand(nQ1)
    J1term_rhorand_Q1 = 2. * sc.jv(1, Qrhorand_Q1) / Qrhorand_Q1
    test_Kz_Q1Q2 = Vcylinder * J1term_rhorand_Q1[:, np.newaxis]\
        * test_Kz_Q2[np.newaxis, :]

    # Cylinder radii to check
    nr = 5
    rc_r = 3. * np.random.rand(nr)

    # Cylinder height to check
    nh = 7
    hc_h = 4. * np.random.rand(nh)

    # Wave vector directions in-plane to check
    nd = 13
    Qrho_dv = np.zeros((nd, 3), dtype=float)
    Qrho_dv[:, :2] = 2. * np.random.rand(nd, 2) - 1.
    Qrho_dv /= np.linalg.norm(Qrho_dv, axis=1)[:, np.newaxis]  # normalize

    # ---------- Script ---------- #

    for rc in rc_r:
        for hc in hc_h:
            # Set up wave vectors for radial tests
            Qrho_dQ1v = Qrhorel_Q1[np.newaxis, :, np.newaxis] / rc\
                * Qrho_dv[:, np.newaxis, :]
            Qrho_Q2v = Qzrand_Q2[:, np.newaxis] / (hc / 2.)\
                * np.array([0., 0., 1.])[np.newaxis, :]
            Qrho_dQ1Q2v = Qrho_dQ1v[..., np.newaxis,
                                    :] + Qrho_Q2v[np.newaxis, np.newaxis, ...]

            # Set up wave vectors for cylindrical tests
            Qz_dQ1v = Qrhorand_Q1[np.newaxis, :, np.newaxis] / rc\
                * Qrho_dv[:, np.newaxis, :]
            Qz_Q2v = Qzrel_Q2[:, np.newaxis] / (hc / 2.)\
                * np.array([0., 0., 1.])[np.newaxis, :]
            Qz_dQ1Q2v = Qz_dQ1v[..., np.newaxis,
                                :] + Qz_Q2v[np.newaxis, np.newaxis, ...]

            # Calculate geometry factors
            Krho_dQ1Q2 = cylindrical_geometry_factor(Qrho_dQ1Q2v, rc, hc)
            Kz_dQ1Q2 = cylindrical_geometry_factor(Qz_dQ1Q2v, rc, hc)

            # Check against expected result
            assert np.allclose(Krho_dQ1Q2, rc**2. * hc / 2.
                               * test_Krho_Q1Q2[np.newaxis, ...],
                               atol=1.e-8)
            assert np.allclose(Kz_dQ1Q2, rc**2. * hc / 2.
                               * test_Kz_Q1Q2[np.newaxis, ...],
                               atol=1.e-8)


def parallelepipedic_kernel_test():
    pass


def Co_hcp_test():
    """Check that the site kernel interface works on run time inputs.

    To do: Remember to check full diagonal as well as hermitian properties XXX
    """
    # ---------- Inputs ---------- #

    # Part 1: Generate plane wave representation (PWDescriptor)
    # Atomic configuration
    a = 2.5071
    c = 4.0695
    mm = 1.6
    # Ground state settings
    xc = 'LDA'
    kpts = 4
    pw = 200
    # Response settings
    ecut = 50.
    gammacentered = False
    q_c = [0., 0., 0.]

    # Part 2: Calculate site kernels
    rc_m = np.array([2.0, 3.0])  # radii in Å
    zc_m = ['diameter', 'unit cell']

    # Part 3: Check the calculated kernels

    # ---------- Script ---------- #

    # Part 1: Generate plane wave representation (PWDescriptor)
    atoms = bulk('Co', 'hcp', a=a, c=c)
    atoms.set_initial_magnetic_moments([mm, mm])

    calc = GPAW(xc=xc,
                spinpol=True,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts),
                      'gamma': True}
                )

    pd0 = get_PWDescriptor(atoms, calc, q_c,
                           ecut=ecut,
                           gammacentered=gammacentered)

    # Part 2: Calculate site kernels
    siteposition_mv = atoms.get_positions()

    # Compute site-kernels
    # How can it make sense to use a paralleliped for Co(hcp)?  XXX
    Kuc_GGm = site_kernel_interface(pd0, siteposition_mv,
                                    shapes_m='unit cell')
    Ksph_GGm = site_kernel_interface(pd0, siteposition_mv,
                                     shapes_m='sphere', rc_m=rc_m)
    Kcyl_GGm = site_kernel_interface(pd0, siteposition_mv,
                                     shapes_m='cylinder', rc_m=rc_m, zc_m=zc_m)

    # Part 4: Check the calculated kernels
    nG = len(get_pw_coordinates(pd0))
    V0 = atoms.get_volume()  # Volume of unit cell in Å^3

    # Calculate integration volumes in Å^3
    Vsphere_m = 4 / 3 * np.pi * rc_m**3
    height_m = np.array([2 * rc_m[0], np.sum(atoms.cell[:, -1])])
    Vcylinder_m = np.pi * rc_m**2 * height_m

    # Check shape of K-arrays
    assert Kuc_GGm.shape == (nG, nG, 1)
    assert Ksph_GGm.shape == (nG, nG, 2)
    assert Kcyl_GGm.shape == (nG, nG, 2)

    # Check that K_00(q=0) gives Vint / V0 (fractional integration volume)
    assert abs(Kuc_GGm[0, 0, 0] - 1.) < 1.e-8
    assert np.allclose(Ksph_GGm[0, 0, :], Vsphere_m / V0)
    assert np.allclose(Kcyl_GGm[0, 0, :], Vcylinder_m / V0)


# ---------- Test functionality ---------- #


def get_PWDescriptor(atoms, calc, q_c, ecut=50., gammacentered=False):
    """Mock-up of PlaneWaveKSLRF.get_PWDescriptor working on a bare calculator
    instance without any actual data in it."""
    from ase.units import Ha
    from gpaw.pw.descriptor import PWDescriptor
    from gpaw.kpt_descriptor import KPointDescriptor

    # Perform inexpensive calculator initialization
    calc.initialize(atoms)

    # Create the plane wave descriptor
    q_c = np.asarray(q_c, dtype=float)
    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(ecut / Ha, calc.wfs.gd,
                      complex, qd, gammacentered=gammacentered)

    return pd
