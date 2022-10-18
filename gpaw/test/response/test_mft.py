"""Calculate the Heisenberg exchange constants in Fe and Co using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""

# General modules
import pytest
import numpy as np

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw import mpi

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKS
from gpaw.response.localft import (LocalFTCalculator, LocalPAWFTCalculator,
                                   add_magnetization)
from gpaw.response.mft import IsotropicExchangeCalculator
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_fm_magnon_energies)


@pytest.mark.response
def test_Fe_bcc(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 50
    eta = 0.
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

    calc = GPAW(gpw_files['fe_pw_wfs'])
    nbands = calc.parameters.convergence['bands']
    atoms = calc.atoms

    # MFT calculation

    # Set up site kernels with a single site
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    sitekernels.append(CylindricalSiteKernels(positions, ez_pav,
                                              rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    sitekernels.append(ParallelepipedicSiteKernels(positions,
                                                   [[atoms.get_cell()]]))

    # Initialize the exchange calculator
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks = ChiKS(gs, context,
                  ecut=ecut, nbands=nbands, eta=eta,
                  gammacentered=True)
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context)
    isoexch_calc = IsotropicExchangeCalculator(chiks, localft_calc)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)
    Jcorr_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)
        Jcorr_qabp[q] = isoexch_calc(q_c, sitekernels, goldstone_corr=True)
        if np.allclose(q_c, 0.):
            # Make sure that the correction is working as intended
            pd0, chiksr0_GG = isoexch_calc.get_chiksr(np.array([0., 0., 0.]))
            m_G = isoexch_calc.localft_calc(pd0, add_magnetization)
            Bxc_G = isoexch_calc.get_Bxc()
            chiksr0_GG = chiksr0_GG + isoexch_calc.get_goldstone_correction()
            mchi_G = 2. * chiksr0_GG @ Bxc_G
            assert np.allclose(m_G, mchi_G)

    # Since we only have a single site, reduce the array
    J_qp = J_qabp[:, 0, 0, :]
    Jcorr_qp = Jcorr_qabp[:, 0, 0, :]

    # Calculate the magnon energies
    mm = 2.21
    mm_ap = mm * np.ones((1, npartitions))  # Magnetic moments
    mw_qp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)[:, 0, :]
    mwcorr_qp = calculate_fm_magnon_energies(Jcorr_qabp, q_qc,
                                             mm_ap)[:, 0, :]

    # Part 3: Compare results to test values
    test_J_pq = np.array([[2.0232, 1.0577, 1.4658],
                          [2.4102, 1.1007, 1.6106],
                          [6.2476, 0.2641, 1.7574],
                          [1.4543, 0.7550, 1.0597],
                          [2.2350, 1.0943, 1.5223],
                          [4.9430, 0.3441, 1.5484],
                          [2.3273, 1.0998, 1.6097]])
    test_Jcorr_pq = np.array([[2.1802, 1.1497, 1.5866],
                              [2.6004, 1.1965, 1.7416],
                              [6.7616, 0.3008, 1.9197],
                              [1.5718, 0.8226, 1.1545],
                              [2.4140, 1.2004, 1.6514],
                              [5.3270, 0.3732, 1.6703],
                              [2.5145, 1.1922, 1.7434]])
    test_mw_pq = np.array([[0., 0.8738, 0.5044],
                           [0., 1.1851, 0.7236],
                           [0., 5.4149, 4.0636],
                           [0., 0.6329, 0.3571],
                           [0., 1.0322, 0.6450],
                           [0., 4.1619, 3.0720],
                           [0., 1.1108, 0.6494]])
    test_mwcorr_pq = np.array([[0., 0.9326, 0.5372],
                               [0., 1.2706, 0.7773],
                               [0., 5.8469, 4.3819],
                               [0., 0.6779, 0.3776],
                               [0., 1.0983, 0.6901],
                               [0., 4.4831, 3.3092],
                               [0., 1.1966, 0.6978]])

    # Exchange constants
    assert J_qp.imag == pytest.approx(0.0)
    assert J_qp.real.T == pytest.approx(test_J_pq, rel=2e-3)
    assert Jcorr_qp.imag == pytest.approx(0.0)
    assert Jcorr_qp.real.T == pytest.approx(test_Jcorr_pq, rel=2e-3)

    # Magnon energies
    assert mw_qp.T == pytest.approx(test_mw_pq, rel=2e-3)
    assert mwcorr_qp.T == pytest.approx(test_mwcorr_pq, rel=2e-3)


@pytest.mark.response
@pytest.mark.skipif(mpi.size == 1, reason='Slow test, skip in serial')
def test_Co_hcp(in_tmp_dir):
    # ---------- Inputs ---------- #

    # Part 1: Ground state calculation
    # Atomic configuration
    a = 2.5071
    c = 4.0695
    mm = 1.6
    # Ground state parameters
    xc = 'LDA'
    kpts = 6
    occw = 0.01
    nbands = 2 * (6 + 0)  # 4s + 3d + 0 empty shell bands
    ebands = 2 * 2  # extra bands for ground state calculation
    pw = 200
    conv = {'density': 1e-8,
            'forces': 1e-8,
            'bands': nbands}

    # Part 2: MFT calculation
    ecut = 100
    eta0 = 0.
    eta1 = 0.1
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [1. / 3., 1 / 3., 0.],  # K
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

    # Part 3: Compare results to test values
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

    # Part 1: Ground state calculation

    atoms = bulk('Co', 'hcp', a=a, c=c)
    atoms.set_initial_magnetic_moments([mm, mm])
    atoms.center()

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                occupations=FermiDirac(occw),
                convergence=conv,
                nbands=nbands + ebands,
                parallel={'domain': 1})

    atoms.calc = calc
    atoms.get_potential_energy()

    # Part 2: MFT calculation

    # Set up spherical site kernels
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)

    # Set up a site kernel to fill out the entire unit cell
    cell_cv = atoms.get_cell()
    cc_v = np.sum(cell_cv, axis=0) / 2.  # Unit cell center
    ucsitekernels = ParallelepipedicSiteKernels([cc_v], [[cell_cv]])

    # Initialize the exchange calculator with and without eta,
    # as well as with and without symmetry
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks0 = ChiKS(gs, context,
                   disable_point_group=True,
                   disable_time_reversal=True,
                   ecut=ecut, nbands=nbands, eta=eta0,
                   gammacentered=True)
    localft_calc = LocalPAWFTCalculator(gs, context)
    isoexch_calc0 = IsotropicExchangeCalculator(chiks0, localft_calc)
    chiks1 = ChiKS(gs, context,
                   ecut=ecut, nbands=nbands, eta=eta1,
                   gammacentered=True)
    isoexch_calc1 = IsotropicExchangeCalculator(chiks1, localft_calc)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)
    Jcorr_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_qe = np.empty((nq, 2), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc0(q_c, sitekernels)
        Jcorr_qabp[q] = isoexch_calc0(q_c, sitekernels, goldstone_corr=True)
        if np.allclose(q_c, 0.):
            # Make sure that the correction is working as intended
            pd0, chiksr0_GG = isoexch_calc0.get_chiksr(np.array([0., 0., 0.]))
            m_G = isoexch_calc0.localft_calc(pd0, add_magnetization)
            Bxc_G = isoexch_calc0.get_Bxc()
            chiksr0_GG = chiksr0_GG + isoexch_calc0.get_goldstone_correction()
            mchi_G = 2. * chiksr0_GG @ Bxc_G
            assert np.allclose(m_G, mchi_G)
        Juc_qe[q, 0] = isoexch_calc0(q_c, ucsitekernels)[0, 0, 0]
        Juc_qe[q, 1] = isoexch_calc1(q_c, ucsitekernels)[0, 0, 0]

    # Calculate the magnon energy
    mm_ap = calc.get_magnetic_moment() / 2.\
        * np.ones((nsites, npartitions))
    mw_qnp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwcorr_qnp = calculate_fm_magnon_energies(Jcorr_qabp, q_qc, mm_ap)
    mwcorr_qnp = np.sort(mwcorr_qnp, axis=1)
    mwuc_qe = calculate_single_site_magnon_energies(Juc_qe, q_qc,
                                                    calc.get_magnetic_moment())

    # Part 3: Compare results to test values
    test_J_qab = np.array([[[1.37280847 + 0.j, 0.28516320 + 0.00007375j],
                            [0.28516320 - 0.00007375j, 1.37280847 - 0.j]],
                           [[0.99649489 - 0.j, 0.08201540 - 0.04905246j],
                            [0.08201540 + 0.04905246j, 0.99649489 + 0.j]],
                           [[0.95009010 + 0.j, -0.0329297 - 0.05777656j],
                            [-0.0329297 + 0.05777656j, 0.95009010 + 0.j]],
                           [[1.30186322 - 0.j, 0.00000038 - 0.00478552j],
                            [0.00000038 + 0.00478552j, 1.30186322 - 0.j]]])
    test_Jcorr_qab = np.array([[[1.38488443 + 0.j, 0.297239160 + 0.j],
                                [0.29723916 + 0.j, 1.384884430 + 0.j]],
                               [[1.00845260 + 0.j, 0.088034249 - 0.05938509j],
                                [0.08803425 + 0.05938509j, 1.00845260 + 0.j]],
                               [[0.96189789 + 0.j, -0.03881925 - 0.06801072j],
                                [-0.03881925 + 0.06801072j, 0.96189789 + 0.j]],
                               [[1.31359555 + 0.j, 0.00000038 - 0.01651785j],
                                [0.00000038 + 0.01651785j, 1.31359555 + 0.j]]])
    test_mw_qn = np.array([[0., 0.673172311],
                           [0.667961643, 0.893557698],
                           [0.757038564, 0.914026524],
                           [0.414677028, 0.425972649]])
    test_mwcorr_qn = np.array([[0., 0.701679491],
                               [0.669812099, 0.920493548],
                               [0.757671646, 0.942533678],
                               [0.415487530, 0.454480499]])
    test_mwuc_q = np.array([0., 0.72440073, 1.2123005, 0.37567975])

    # Exchange constants
    # err = np.absolute(J_qabp[..., 1] - test_J_qab)
    # is_bad = err > J_atol + J_rtol * np.absolute(test_J_qab)
    # print(is_bad)
    # print(np.absolute(err[is_bad] / np.absolute(test_J_qab[is_bad])))
    assert np.allclose(J_qabp[..., 1], test_J_qab,
                       atol=J_atol, rtol=J_rtol)
    assert np.allclose(Jcorr_qabp[..., 1], test_Jcorr_qab,
                       atol=J_atol, rtol=J_rtol)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.all(np.abs(mwcorr_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.allclose(mwuc_qe[0, :], 0.)  # Goldstone
    assert np.allclose(mw_qnp[1:, 0, 1], test_mw_qn[1:, 0], rtol=mw_rtol)
    assert np.allclose(mw_qnp[:, 1, 1], test_mw_qn[:, 1], rtol=mw_rtol)
    assert np.allclose(mwcorr_qnp[1:, 0, 1], test_mwcorr_qn[1:, 0],
                       rtol=mw_rtol)
    assert np.allclose(mwcorr_qnp[:, 1, 1], test_mwcorr_qn[:, 1],
                       rtol=mw_rtol)
    assert np.allclose(mwuc_qe[1:, 0], test_mwuc_q[1:], rtol=mw_rtol)

    # Part 4: Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2],
                       test_mw_qn[1:, 0, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mw_qnp[:, 1, ::2],
                       test_mw_qn[:, 1, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mwcorr_qnp[1:, 0, ::2],
                       test_mwcorr_qn[1:, 0, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mwcorr_qnp[:, 1, ::2],
                       test_mwcorr_qn[:, 1, np.newaxis], rtol=mw_ctol)
    # Check that a finite eta does not change the magnon energies too much
    assert np.allclose(mwuc_qe[1:, 0], mwuc_qe[1:, 1], rtol=mw_ctol)
