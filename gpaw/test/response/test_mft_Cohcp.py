"""Calculate the Heisenberg exchange constants in Co(hcp) using MFT.
Test numerics and consistency of acoustic branch with a unit cell kernel."""

# General modules
import numpy as np

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.mft import IsotropicExchangeCalculator
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_FM_magnon_energies)


def test_Co_hcp():
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
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [1. / 3., 1 / 3., 0.],  # K
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

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
                idiotproof=False,
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

    # Initialize the exchange calculator
    isoexch_calc = IsotropicExchangeCalculator(calc, ecut=ecut, nbands=nbands)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_q = np.empty((nq,), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)
        Juc_q[q] = isoexch_calc(q_c, ucsitekernels)[0, 0, 0]

    # Calculate the magnon energy
    mm_ap = calc.get_magnetic_moment() / 2.\
        * np.ones((nsites, npartitions))
    mw_qnp = calculate_FM_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwuc_q = calculate_single_site_magnon_energies(Juc_q, q_qc,
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
    test_mw_qn = np.array([[0., 0.673172482],
                           [0.668238523, 0.893387671],
                           [0.757884234, 0.913272672],
                           [0.414111193, 0.426511947]])
    test_mwuc_q = np.array([0., 0.72445911, 1.21301805, 0.37568663])

    # Exchange constants
    assert np.allclose(J_qabp[..., 1], test_J_qab, rtol=1e-3)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert abs(mwuc_q[0]) < 1.e-8  # Goldstone
    assert np.allclose(mw_qnp[1:, 0, 1], test_mw_qn[1:, 0], rtol=1.e-3)
    assert np.allclose(mw_qnp[:, 1, 1], test_mw_qn[:, 1], rtol=1.e-3)
    assert np.allclose(mwuc_q[1:], test_mwuc_q[1:], rtol=1.e-3)

    # Part 4: Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2], test_mw_qn[1:, 0, np.newaxis],
                       rtol=5.e-2)
    assert np.allclose(mw_qnp[:, 1, ::2], test_mw_qn[:, 1, np.newaxis],
                       rtol=5.e-2)
