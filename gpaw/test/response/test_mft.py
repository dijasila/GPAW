"""Calculate the Heisenberg exchange constants in Fe and Co using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""

# General modules
import pytest
import numpy as np

# Script modules
from ase.units import Bohr

from gpaw import GPAW
from gpaw.sphere.integrate import integrate_lebedev

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator, smat
from gpaw.response.localft import (LocalFTCalculator, LocalPAWFTCalculator,
                                   add_spin_polarization)
from gpaw.response.mft import (IsotropicExchangeCalculator, AtomicSiteData,
                               SumRuleSiteMagnetization,
                               SumRuleSiteMagnetizationCalculator)
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_fm_magnon_energies)
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.pair_transitions import PairTransitions
from gpaw.response.matrix_elements import SitePairDensityCalculator
from gpaw.test.conftest import response_band_cutoff
from gpaw.test.response.test_chiks import generate_qrel_q, get_q_c


@pytest.mark.response
def test_Fe_bcc(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 50
    # Do the high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     ])
    # Define site kernels to test
    # Test a single site of spherical and cylindrical geometries
    rc_pa = np.array([[1.0], [1.5], [2.0]])
    hc_pa = np.array([[1.0], [1.5], [2.0]])
    ez_pav = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['fe_pw']
    atoms = calc.atoms

    # Set up site kernels with a single site
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    sitekernels.append(CylindricalSiteKernels(positions, ez_pav,
                                              rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    sitekernels.append(ParallelepipedicSiteKernels(positions,
                                                   [[atoms.get_cell()]]))

    # Initialize the Heisenberg exchange calculator
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc = ChiKSCalculator(gs, context,
                                 ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context)
    isoexch_calc = IsotropicExchangeCalculator(chiks_calc, localft_calc)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)

    # Since we only have a single site, reduce the array
    J_qp = J_qabp[:, 0, 0, :]

    # Calculate the magnon energies
    mm = 2.21
    mm_ap = mm * np.ones((1, npartitions))  # Magnetic moments
    mw_qp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)[:, 0, :]

    # Compare results to test values
    test_J_pq = np.array(
        [[2.1907596825086455, 1.172424411323134, 1.6060583789867644],
         [2.612428039019977, 1.2193926800088601, 1.7635196888465006],
         [6.782367391186284, 0.2993922109834177, 1.9346016211386057],
         [1.5764800860123762, 0.8365204592352894, 1.1648584638500161],
         [2.4230224513213234, 1.2179759558303274, 1.6691805687218078],
         [5.35668502504496, 0.3801778545994659, 1.6948968244858478],
         [2.523580017606111, 1.21779750159267, 1.7637120466695273]])
    test_mw_pq = np.array(
        [[0.0, 0.9215703811633589, 0.5291414511510236],
         [0.0, 1.2606654832679791, 0.7682428508357253],
         [0.0, 5.866945864436984, 4.38711834393455],
         [0.0, 0.6696467210652369, 0.3725082553505521],
         [0.0, 1.0905398149239784, 0.682209848506349],
         [0.0, 4.503626398593207, 3.313835475619106],
         [0.0, 1.181703634401304, 0.6876633221145555]])

    # Exchange constants
    assert J_qp.imag == pytest.approx(0.0)
    assert J_qp.T.real == pytest.approx(test_J_pq, rel=2e-3)

    # Magnon energies
    assert mw_qp.T == pytest.approx(test_mw_pq, rel=2e-3)


@pytest.mark.response
def test_Co_hcp(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 100
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

    # Unfortunately, the usage of symmetry leads to such extensive repetition
    # of random noise, that one cannot trust individual values of J very well.
    # This is improved when increasing the number of k-points, but the problem
    # never completely vanishes
    J_atol = 5.e-3
    J_rtol = 5.e-2
    # However, derived physical values have an increased error cancellation due
    # to their collective nature.
    mw_rtol = 5.e-3  # relative tolerance of absolute results
    mw_ctol = 5.e-2  # relative tolerance on kernel and eta self-consistency

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    atoms = calc.get_atoms()

    # Set up spherical site kernels
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)

    # Set up a site kernel to fill out the entire unit cell
    cell_cv = atoms.get_cell()
    cc_v = np.sum(cell_cv, axis=0) / 2.  # Unit cell center
    ucsitekernels = ParallelepipedicSiteKernels([cc_v], [[cell_cv]])

    # Initialize the exchange calculator with and without symmetry
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc0 = ChiKSCalculator(gs, context,
                                  disable_point_group=True,
                                  disable_time_reversal=True,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalPAWFTCalculator(gs, context)
    isoexch_calc0 = IsotropicExchangeCalculator(chiks_calc0, localft_calc)
    chiks_calc1 = ChiKSCalculator(gs, context,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    isoexch_calc1 = IsotropicExchangeCalculator(chiks_calc1, localft_calc)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_qs = np.empty((nq, 2), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc0(q_c, sitekernels)
        chiksr_buffer = isoexch_calc0._chiksr
        Juc_qs[q, 0] = isoexch_calc0(q_c, ucsitekernels)[0, 0, 0]
        assert isoexch_calc0._chiksr is chiksr_buffer, \
            'Two subsequent IsotropicExchangeCalculator calls with the same '\
            'q_c, should reuse, not update, the chiks buffer'

        Juc_qs[q, 1] = isoexch_calc1(q_c, ucsitekernels)[0, 0, 0]

    # Calculate the magnon energy
    mom = atoms.get_magnetic_moment()
    mm_ap = mom / 2.0 * np.ones((nsites, npartitions))
    mw_qnp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwuc_qs = calculate_single_site_magnon_energies(Juc_qs, q_qc, mom)

    # Compare results to test values
    # print(J_qabp[..., 1])
    # print(mw_qnp[..., 1])
    # print(mwuc_qs[:, 0])
    test_J_qab = np.array([[[1.23106207 - 0.j, 0.25816335 - 0.j],
                            [0.25816335 + 0.j, 1.23106207 + 0.j]],
                           [[0.88823839 + 0.j, 0.07345416 - 0.04947835j],
                            [0.07345416 + 0.04947835j, 0.88823839 + 0.j]],
                           [[1.09349955 - 0.j, 0.00000010 - 0.01176761j],
                            [0.00000010 + 0.01176761j, 1.09349955 - 0.j]]])
    test_mw_qn = np.array([[0., 0.64793939],
                           [0.64304039, 0.86531921],
                           [0.48182997, 0.51136436]])
    test_mwuc_q = np.array([0., 0.69678659, 0.44825874])

    # Exchange constants
    # err = np.absolute(J_qabp[..., 1] - test_J_qab)
    # is_bad = err > J_atol + J_rtol * np.absolute(test_J_qab)
    # print(is_bad)
    # print(np.absolute(err[is_bad] / np.absolute(test_J_qab[is_bad])))
    assert np.allclose(J_qabp[..., 1], test_J_qab,
                       atol=J_atol, rtol=J_rtol)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.allclose(mwuc_qs[0, :], 0.)  # Goldstone
    assert np.allclose(mw_qnp[1:, 0, 1], test_mw_qn[1:, 0], rtol=mw_rtol)
    assert np.allclose(mw_qnp[:, 1, 1], test_mw_qn[:, 1], rtol=mw_rtol)
    assert np.allclose(mwuc_qs[1:, 0], test_mwuc_q[1:], rtol=mw_rtol)

    # Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2],
                       test_mw_qn[1:, 0, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mw_qnp[:, 1, ::2],
                       test_mw_qn[:, 1, np.newaxis], rtol=mw_ctol)
    # Check that symmetry toggle do not change the magnon energies
    assert np.allclose(mwuc_qs[1:, 0], mwuc_qs[1:, 1], rtol=mw_ctol)


@pytest.mark.response
def test_Fe_site_magnetization(gpw_files):
    # Set up ground state adapter
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)

    # Extract valid site radii range
    rmin_a, rmax_a = AtomicSiteData.valid_site_radii_range(gs)
    rmin = rmin_a[0]  # Only one magnetic atom in the unit cell
    rmax = rmax_a[0]
    # We expect rmax to be equal to the nearest neighbour distance
    # subtracted with the augmentation sphere radius. For a bcc lattice,
    # nn_dist = sqrt(3) a / 2:
    augr = gs.get_aug_radii()[0]
    rmax_expected = np.sqrt(3) * 2.867 / 2. - augr * Bohr
    assert abs(rmax - rmax_expected) < 1e-6
    # Test that an error is raised outside the valid range
    with pytest.raises(AssertionError):
        AtomicSiteData(gs, indices=[0],  # Too small radii
                       radii=[np.linspace(rmin * 0.8, rmin, 5)])
    with pytest.raises(AssertionError):
        AtomicSiteData(gs, indices=[0],  # Too large radii
                       radii=[np.linspace(rmax, rmax * 1.2, 5)])
    # Define atomic sites to span the valid range
    rc_r = np.linspace(rmin_a[0], rmax_a[0], 100)
    # Add the radius of the augmentation sphere explicitly
    rc_r = np.append(rc_r, [augr * Bohr])
    atomic_sites = AtomicSiteData(gs, indices=[0], radii=[rc_r])

    # Calculate site magnetization
    magmom_ar = atomic_sites.calculate_magnetic_moments()
    magmom_r = magmom_ar[0]

    # Test that a cutoff at the augmentation sphere radius reproduces
    # the local magnetic moment of the GPAW calculation
    magmom_at_augr = calc.get_atoms().get_magnetic_moments()[0]
    assert abs(magmom_r[-1] - magmom_at_augr) < 1e-2

    # Do a manual calculation of the magnetic moment using the
    # all-electron partial waves
    # Calculate all-electron m(r)
    microsetup = atomic_sites.microsetup_a[0]
    m_ng = np.array([microsetup.rgd.zeros()
                     for n in range(microsetup.Y_nL.shape[0])])
    for n, Y_L in enumerate(microsetup.Y_nL):
        n_sg = np.dot(Y_L, microsetup.n_sLg)
        add_spin_polarization(microsetup.rgd, n_sg, m_ng[n, :])
    # Integrate with varrying radii
    m_g = integrate_lebedev(m_ng)
    ae_magmom_r = np.array([
        microsetup.rgd.integrate_trapz(m_g, rcut=rcut / Bohr)
        for rcut in rc_r])
    # Test that values match approximately inside the augmentation sphere
    inaug_r = rc_r <= augr * Bohr
    assert magmom_r[inaug_r] == pytest.approx(ae_magmom_r[inaug_r], abs=3e-2)

    # import matplotlib.pyplot as plt
    # plt.plot(rc_r[:-1], magmom_r[:-1])
    # plt.plot(rc_r[:-1], ae_magmom_r[:-1], zorder=0)
    # plt.axvline(augr * Bohr, c='0.5', linestyle='--')
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.show()


@pytest.mark.response
def test_Co_site_data(gpw_files):
    # Set up ground state adapter
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)

    # Extract valid site radii range
    rmin_a, rmax_a = AtomicSiteData.valid_site_radii_range(gs)
    # The valid ranges should be equal due to symmetry
    assert abs(rmin_a[1] - rmin_a[0]) < 1e-8
    assert abs(rmax_a[1] - rmax_a[0]) < 1e-8
    rmin = rmin_a[0]
    rmax = rmax_a[0]
    # We expect rmax to be equal to the nearest neighbour distance
    # subtracted with the augmentation sphere radius. For the hcp-lattice,
    # nn_dist = min(a, sqrt(a^2/3 + c^2/4)):
    augr_a = gs.get_aug_radii()
    assert abs(augr_a[1] - augr_a[0]) < 1e-8
    augr = augr_a[0]
    rmax_expected = min(2.5071, np.sqrt(2.5071**2 / 3 + 4.0695**2 / 4))
    rmax_expected -= augr * Bohr
    assert abs(rmax - rmax_expected) < 1e-6

    # Use radii spanning the entire valid range
    rc_r = np.linspace(rmin, rmax, 101)
    # Add the radius of the augmentation sphere explicitly
    rc_r = np.append(rc_r, [augr * Bohr])
    nr = len(rc_r)
    # Varry the site radii together and independently
    rc1_r = list(rc_r) + list(rc_r) + [augr * Bohr] * nr
    rc2_r = list(rc_r) + [augr * Bohr] * nr + list(rc_r)
    atomic_sites = AtomicSiteData(gs, indices=[0, 1], radii=[rc1_r, rc2_r])

    # Calculate site magnetization
    magmom_ar = atomic_sites.calculate_magnetic_moments()

    # Test that the magnetization inside the augmentation sphere matches
    # the local magnetic moment of the GPAW calculation
    magmom_at_augr_a = calc.get_atoms().get_magnetic_moments()
    assert magmom_ar[:, -1] == pytest.approx(magmom_at_augr_a, abs=2e-2)

    # Test consistency of varrying radii
    assert magmom_ar[0, :nr] == pytest.approx(magmom_ar[1, :nr])
    assert magmom_ar[0, nr:2 * nr] == pytest.approx(magmom_ar[0, :nr])
    assert magmom_ar[0, 2 * nr:] == pytest.approx([magmom_ar[0, -1]] * nr)
    assert magmom_ar[1, nr:2 * nr] == pytest.approx([magmom_ar[1, -1]] * nr)
    assert magmom_ar[1, 2 * nr:] == pytest.approx(magmom_ar[1, :nr])

    # Calculate the atomic spin splitting
    rc_r = rc_r[:-1]
    atomic_sites = AtomicSiteData(gs, indices=[0, 1], radii=[rc_r, rc_r])
    dxc_ar = atomic_sites.calculate_spin_splitting()
    print(dxc_ar[0, ::20])

    # Test that the spin splitting comes out as expected
    assert dxc_ar[0] == pytest.approx(dxc_ar[1])
    assert dxc_ar[0, ::20] == pytest.approx([0.02638351, 1.41476112,
                                             2.49540004, 2.79727200,
                                             2.82727948, 2.83670767], rel=1e-3)

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.plot(rc_r, magmom_ar[0, :nr - 1])
    # plt.axvline(augr * Bohr, c='0.5', linestyle='--')
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.subplot(1, 2, 2)
    # plt.plot(rc_r, dxc_ar[0])
    # plt.axvline(augr * Bohr, c='0.5', linestyle='--')
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$\Delta_\mathrm{xc}$ [eV]')
    # plt.show()


@pytest.mark.response
@pytest.mark.parametrize('qrel', generate_qrel_q())
def test_Co_site_magnetization_sum_rule(in_tmp_dir, gpw_files, qrel):
    # Set up ground state adapter
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext('Co_sum_rule.txt')

    if context.comm.size % 4 == 0:
        nblocks = 4
    elif context.comm.size % 2 == 0:
        nblocks = 2
    else:
        nblocks = 1

    # Get wave vector to test
    q_c = get_q_c('co_pw', qrel)

    # Set up atomic sites
    rmin_a, _ = AtomicSiteData.valid_site_radii_range(gs)
    # Make sure that the two sites do not overlap
    nn_dist = min(2.5071, np.sqrt(2.5071**2 / 3 + 4.0695**2 / 4))
    rc_r = np.linspace(rmin_a[0], nn_dist / 2, 11)
    atomic_site_data = AtomicSiteData(gs, indices=[0, 1], radii=[rc_r, rc_r])

    # ----- Site magnetization from site pair densities ----- #
    # Set up calculator and calculate the site magnetization
    simple_site_mag_calc = SimpleSiteMagnetizationCalculator(gs, context)
    ssite_mag_ar = simple_site_mag_calc(atomic_site_data)

    # Test that the imaginary part vanishes (we use only diagonal pair
    # densities correcsponding to |ψ_nks(r)|^2)
    assert np.allclose(ssite_mag_ar.imag, 0.)
    ssite_mag_ar = ssite_mag_ar.real

    # Test that the results match a conventional calculation
    magmom_ar = atomic_site_data.calculate_magnetic_moments()
    assert ssite_mag_ar == pytest.approx(magmom_ar, rel=5e-3)

    # ----- Site magnetization by sum rule ----- #
    # Set up calculator and calculate site magnetization by sum rule
    sum_rule_site_mag_calc = SumRuleSiteMagnetizationCalculator(
        gs, context, nblocks=nblocks, nbands=nbands)
    site_mag_abr = sum_rule_site_mag_calc(q_c, atomic_site_data)
    context.write_timer()

    # Test that the sum rule site magnetization is a positive-valued diagonal
    # real array
    site_mag_ra = site_mag_abr.diagonal()
    assert np.all(site_mag_ra.real > 0)
    assert np.all(np.abs(site_mag_ra.imag) / site_mag_ra.real < 1e-6)
    site_mag_ra = site_mag_ra.real
    assert np.all(np.abs(np.diagonal(np.fliplr(  # off-diagonal elements
        site_mag_abr))) / site_mag_ra < 5e-2)
    site_mag_ar = site_mag_ra.T
    # Test that the magnetic moments on the two Co atoms are identical
    assert site_mag_ar[0] == pytest.approx(site_mag_ar[1], rel=1e-4)

    # Test that the result more or less matches a conventional calculation at
    # close-packing
    assert np.average(site_mag_ar, axis=0)[-1] == pytest.approx(
        np.average(magmom_ar, axis=0)[-1], rel=5e-2)

    # Test values against reference
    print(np.average(site_mag_ar, axis=0)[::2])
    assert np.average(site_mag_ar, axis=0)[::2] == pytest.approx(
        np.array([3.91823444e-04, 1.45641911e-01, 6.85939109e-01,
                  1.18813171e+00, 1.49761591e+00, 1.58954270e+00]), rel=5e-2)

    # import matplotlib.pyplot as plt
    # plt.plot(rc_r, site_mag_ar[0], '-o', mec='k')
    # plt.plot(rc_r, ssite_mag_ar[0], '-o', mec='k', zorder=1)
    # plt.plot(rc_r, magmom_ar[0], '-o', mec='k', zorder=0)
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.title(str(q_c))
    # plt.show()


# ---------- Test functionality ---------- #


class SimpleSiteMagnetization(SumRuleSiteMagnetization):
    @property
    def shape(self):
        nsites = self.atomic_site_data.nsites
        npartitions = self.atomic_site_data.npartitions
        return nsites, npartitions


class SimpleSiteMagnetizationCalculator(PairFunctionIntegrator):
    r"""Calculate the site magnetization using site pair densities.

    The site magnetization can be calculate from the diagonals of the site pair
    densities as follows:
                 __  __
             1   \   \
    n_a^z = ‾‾‾  /   /  σ^z_ss f_nks n^a_(nks,nks)
            N_k  ‾‾  ‾‾
                 k   n,s
    """

    def __init__(self, gs, context):
        super().__init__(gs, context,
                         disable_point_group=True,
                         disable_time_reversal=True)
        self.site_pair_density_calc: SitePairDensityCalculator | None = None

    def __call__(self, atomic_site_data):
        self.site_pair_density_calc = SitePairDensityCalculator(
            self.gs, self.context, atomic_site_data)

        # Set up transitions
        # Loop over bands, which are fully or partially occupied
        nocc2 = self.kptpair_extractor.nocc2
        n_n = list(range(nocc2))
        n_t = np.array(n_n + n_n)
        s_t = np.array([0] * nocc2 + [1] * nocc2)
        transitions = PairTransitions(n1_t=n_t, n2_t=n_t, s1_t=s_t, s2_t=s_t)

        # Set up data object with q=0
        qpd = self.get_pw_descriptor([0., 0., 0.], ecut=1e-3)
        site_mag = SimpleSiteMagnetization(qpd, atomic_site_data)

        # Perform actual calculation
        self._integrate(site_mag, transitions)
        return site_mag.array

    def add_integrand(self, kptpair, weight, site_mag):
        r"""Add the site magnetization integrand of the outer k-point integral.

        With
                   __
                1  \
        n_a^z = ‾  /  (...)_k
                V  ‾‾
                   k

        the integrand has to be multiplied with the cell volume V0:
                     __
                     \
        (...)_k = V0 /  σ^z_ss f_nks n^a_(nks,nks)
                     ‾‾
                     n,s
        """
        # Calculate site pair densties
        site_pair_density = self.site_pair_density_calc(kptpair, site_mag.qpd)
        assert site_pair_density.tblocks.blockcomm.size == 1
        n_tap = site_pair_density.get_global_array()

        # Calculate Pauli matrix factors multiply the occupations
        sigz = smat('z')
        sigz_t = sigz[kptpair.transitions.s1_t, kptpair.transitions.s2_t]
        f_t = kptpair.get_all(kptpair.ikpt1.f_myt)
        sigzf_t = sigz_t * f_t

        # Calculate and add integrand
        site_mag.array[:] += self.gs.volume * weight * np.einsum(
            't, tap -> ap', sigzf_t, n_tap)
