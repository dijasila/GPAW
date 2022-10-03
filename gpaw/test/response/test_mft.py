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
from gpaw.response.mft import (IsotropicExchangeCalculator,
                               LSDABxcCalculator, GoldstoneBxcCalculator)
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_fm_magnon_energies)
from gpaw.test.response.test_site_kernels import get_PWDescriptor
from gpaw.test.response.test_localft import get_inversion_pairs


@pytest.mark.response
def test_Fe_bcc(in_tmp_dir):
    # ---------- Inputs ---------- #

    # Part 1: Ground state calculation
    xc = 'LDA'
    kpts = 4
    nbands = 6
    pw = 200
    occw = 0.01
    conv = {'density': 1e-8,
            'forces': 1e-8,
            'bands': nbands}
    a = 2.867
    mm = 2.21

    # Part 2: MFT calculation
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

    # Part 1: Ground state calculation

    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])
    atoms.center()

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                nbands=nbands + 4,
                occupations=FermiDirac(occw),
                parallel={'domain': 1},
                spinpol=True,
                convergence=conv
                )

    atoms.calc = calc
    atoms.get_potential_energy()

    # Part 2: MFT calculation

    # Set up site kernels with a single site
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    sitekernels.append(CylindricalSiteKernels(positions, ez_pav,
                                              rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    sitekernels.append(ParallelepipedicSiteKernels(positions,
                                                   [[atoms.get_cell()]]))

    # Initialize the exchange calculator and bxc calculators
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks = ChiKS(gs, context,
                  ecut=ecut, nbands=nbands, eta=eta,
                  gammacentered=True)
    isoexch_calc = IsotropicExchangeCalculator(chiks)
    lsda_bxc_calc = LSDABxcCalculator(gs, context)
    gsto_bxc_calc = GoldstoneBxcCalculator(gs, context)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    Jl_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)
    Jg_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        Jl_qabp[q] = isoexch_calc(q_c, sitekernels, lsda_bxc_calc)
        Jg_qabp[q] = isoexch_calc(q_c, sitekernels, gsto_bxc_calc)
    # Since we only have a single site, reduce the array
    Jl_qp = Jl_qabp[:, 0, 0, :]
    Jg_qp = Jg_qabp[:, 0, 0, :]

    # Extract the magnetic xc potential from the calculators
    assert lsda_bxc_calc.in_buffer() and gsto_bxc_calc.in_buffer()
    Bxcl_G = lsda_bxc_calc.from_buffer()
    Bxcg_G = gsto_bxc_calc.from_buffer()

    # Calculate the magnon energies
    mm_ap = mm * np.ones((1, npartitions))  # Magnetic moments
    mwl_qp = calculate_fm_magnon_energies(Jl_qabp, q_qc, mm_ap)[:, 0, :]
    mwg_qp = calculate_fm_magnon_energies(Jg_qabp, q_qc, mm_ap)[:, 0, :]

    # Part 3: Compare results to test values
    test_Jl_pq = np.array([[1.61655323, 0.88149124, 1.10008928],
                           [1.86800734, 0.93735081, 1.23108285],
                           [4.67979867, 0.2004699, 1.28510023],
                           [1.14516166, 0.62140228, 0.78470217],
                           [1.734752, 0.87124284, 1.13880145],
                           [3.82381708, 0.31159032, 1.18094396],
                           [1.79888576, 0.92972442, 1.2054906]])
    test_Jg_pq = np.array([[1.60080254, 0.88368456, 1.08409183],
                           [2.41061506, 0.78447108, 1.05312028],
                           [7.93690142, 0.36354499, 1.57553707],
                           [1.25460343, 0.68691519, 0.85445696],
                           [2.16870230, 1.21354926, 1.15777672],
                           [5.16091847, 0.20213561, 1.02231494],
                           [2.32270945, 0.83722209, 1.22373215]])
    test_mwl_pq = np.array([[0., 0.66521177, 0.46738581],
                            [0., 0.84222041, 0.57640002],
                            [0., 4.05369028, 3.07212255],
                            [0., 0.47398746, 0.32620549],
                            [0., 0.78145439, 0.53931965],
                            [0., 3.17848551, 2.39173871],
                            [0., 0.78656857, 0.5370069]])
    test_mwg_pq = np.array([[0., 0.64897555, 0.46761150],
                            [0., 1.47162352, 1.22850206],
                            [0., 6.85371623, 5.75689082],
                            [0., 0.51374501, 0.36212350],
                            [0., 0.86439189, 0.91486478],
                            [0., 4.48758630, 3.74534255],
                            [0., 1.34433245, 0.99454959]])

    # Symmetry of the magnetic xc potentials
    pd0 = get_PWDescriptor(atoms, calc, [0., 0., 0.],
                           ecut=ecut,
                           gammacentered=True)
    G1_G, G2_G = get_inversion_pairs(pd0)
    assert np.allclose(np.conj(Bxcl_G[G1_G]), Bxcl_G[G2_G])
    assert np.allclose(np.conj(Bxcg_G[G1_G]), Bxcg_G[G2_G])

    # Exchange constants
    assert np.allclose(Jl_qp.imag, 0.)
    assert np.allclose(Jl_qp.real, test_Jl_pq.T, rtol=2e-3)
    assert np.allclose(Jg_qp.imag, 0.)
    assert np.allclose(Jg_qp.real, test_Jg_pq.T, rtol=2e-3)
    
    # Magnon energies
    assert np.allclose(mwl_qp, test_mwl_pq.T, rtol=2e-3)
    assert np.allclose(mwg_qp, test_mwg_pq.T, rtol=2e-3)


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
    isoexch_calc0 = IsotropicExchangeCalculator(chiks0)
    chiks1 = ChiKS(gs, context,
                   ecut=ecut, nbands=nbands, eta=eta1,
                   gammacentered=True)
    isoexch_calc1 = IsotropicExchangeCalculator(chiks1)
    # Initialize the bxc calculator
    bxc_calc = LSDABxcCalculator(gs, context)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_qe = np.empty((nq, 2), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc0(q_c, sitekernels, bxc_calc)
        Juc_qe[q, 0] = isoexch_calc0(q_c, ucsitekernels, bxc_calc)[0, 0, 0]
        Juc_qe[q, 1] = isoexch_calc1(q_c, ucsitekernels, bxc_calc)[0, 0, 0]

    # Calculate the magnon energy
    mm_ap = calc.get_magnetic_moment() / 2.\
        * np.ones((nsites, npartitions))
    mw_qnp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwuc_qe = calculate_single_site_magnon_energies(Juc_qe, q_qc,
                                                    calc.get_magnetic_moment())

    # Part 3: Compare results to test values
    test_J_qab = np.array([[[1.37280875 - 0.j,
                             0.28516328 - 0.00007259j],
                            [0.28516328 + 0.00007259j,
                             1.37280875 - 0.j]],
                           [[0.99644998 + 0.j,
                             0.08202191 - 0.04867163j],
                            [0.08202191 + 0.04867163j,
                             0.99644998 + 0.j]],
                           [[0.95005156 - 0.j,
                             -0.03339854 - 0.05672191j],
                            [-0.03339854 + 0.05672191j,
                             0.950051561 + 0.j]],
                           [[1.30187481 - 0.j,
                             0.00000039 - 0.00525360j],
                            [0.00000039 + 0.00525360j,
                             1.30187481 + 0.j]]])
    test_J_qab = np.array([[[1.37280847 + 0.j, 0.28516320 + 0.00007375j],
                            [0.28516320 - 0.00007375j, 1.37280847 - 0.j]],
                           [[0.99649489 - 0.j, 0.08201540 - 0.04905246j],
                            [0.08201540 + 0.04905246j, 0.99649489 + 0.j]],
                           [[0.95009010 + 0.j, -0.0329297 - 0.05777656j],
                            [-0.0329297 + 0.05777656j, 0.95009010 + 0.j]],
                           [[1.30186322 - 0.j, 0.00000038 - 0.00478552j],
                            [0.00000038 + 0.00478552j, 1.30186322 - 0.j]]])
    test_mw_qn = np.array([[0., 0.673172311],
                           [0.667961643, 0.893557698],
                           [0.757038564, 0.914026524],
                           [0.414677028, 0.425972649]])
    test_mwuc_q = np.array([0., 0.72440073, 1.2123005, 0.37567975])

    # Exchange constants
    # err = np.absolute(J_qabp[..., 1] - test_J_qab)
    # is_bad = err > J_atol + J_rtol * np.absolute(test_J_qab)
    # print(is_bad)
    # print(np.absolute(err[is_bad] / np.absolute(test_J_qab[is_bad])))
    assert np.allclose(J_qabp[..., 1], test_J_qab,
                       atol=J_atol, rtol=J_rtol)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.allclose(mwuc_qe[0, :], 0.)  # Goldstone
    assert np.allclose(mw_qnp[1:, 0, 1], test_mw_qn[1:, 0], rtol=mw_rtol)
    assert np.allclose(mw_qnp[:, 1, 1], test_mw_qn[:, 1], rtol=mw_rtol)
    assert np.allclose(mwuc_qe[1:, 0], test_mwuc_q[1:], rtol=mw_rtol)

    # Part 4: Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2], test_mw_qn[1:, 0, np.newaxis],
                       rtol=mw_ctol)
    assert np.allclose(mw_qnp[:, 1, ::2], test_mw_qn[:, 1, np.newaxis],
                       rtol=mw_ctol)
    # Check that a finite eta does not change the magnon energies too much
    assert np.allclose(mwuc_qe[1:, 0], mwuc_qe[1:, 1], rtol=mw_ctol)
