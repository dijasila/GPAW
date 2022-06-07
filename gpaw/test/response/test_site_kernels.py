"""Test the site kernel calculation functionality of the response code"""
from gpaw import GPAW, PW
from gpaw.response.site_kernels import site_kernel_interface, K_sphere
from gpaw.response.susceptibility import get_pw_coordinates
from ase.build import bulk
import numpy as np


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

    # Expected results (assuming rc=1. scale)
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
        # Calculate spherical kernel with wave vector rescaled after rc
        K_Qd = K_sphere(Q_Qdv / rc, rc)
        # Check against expected result
        assert np.allclose(K_Qd, rc**3. * test_K_Q[:, np.newaxis])


def cylindrical_kernel_test():
    pass


def parallelepipedic_kernel_test():
    pass


def Co_hcp_test():
    """Check that the site kernel interface works on run time inputs."""
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
