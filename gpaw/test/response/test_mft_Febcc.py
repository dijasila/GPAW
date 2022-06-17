"""Calculate the Heisenberg exchange constants in Fe using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""
from gpaw import GPAW, PW, FermiDirac
from ase.build import bulk
import numpy as np
from gpaw.response.mft import IsotropicExchangeCalculator
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import calculate_FM_magnon_energies


def test_Fe_bcc():
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
    # Do the high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     [0.25, 0.25, 0.25]   # P
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
                symmetry={'point_group': False},
                idiotproof=False,
                parallel={'domain': 1},
                spinpol=True,
                convergence=conv
                )

    atoms.calc = calc
    atoms.get_potential_energy()

    # Part 2: MFT calculation

    # Set up single site kernels with a single site
    positions = atoms.get_positions()
    site_kernels = SphericalSiteKernels(positions, rc_pa)
    site_kernels.append(CylindricalSiteKernels(positions, ez_pav,
                                               rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    site_kernels.append(ParallelepipedicSiteKernels(positions,
                                                    [[atoms.get_cell()]]))

    # Initialize the exchange calculator
    exchCalc = IsotropicExchangeCalculator(calc, ecut=ecut, nbands=nbands)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = site_kernels.nsites
    npartitions = site_kernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = exchCalc(q_c, site_kernels)
    # Since we only have a single site, reduce the array
    J_qp = J_qabp[:, 0, 0, :]

    # Calculate the magnon energies
    magmoms_ap = mm * np.ones((1, npartitions))
    mw_qp = calculate_FM_magnon_energies(J_qabp, q_qc, magmoms_ap)[:, 0, :]

    # Run the chiks calculator individually
    chiks_GGq = []  # Could be tested elsewhere? XXX
    for q_c in q_qc:
        _, chiks_GG = exchCalc.chiksf('-+', q_c, txt=None)
        chiks_GGq += [chiks_GG]
    chiks_GGq = np.dstack(chiks_GGq)

    # Part 3: Compare new results to test values
    test_J_pq = np.array([[1.61655323, 0.88149124, 1.10008928, 1.18887259],
                          [1.86800734, 0.93735081, 1.23108285, 1.33289874],
                          [4.67979867, 0.2004699, 1.28510023, 1.30265974],
                          [1.14516166, 0.62140228, 0.78470217, 0.84861897],
                          [1.734752, 0.87124284, 1.13880145, 1.23047167],
                          [3.82381708, 0.31159032, 1.18094396, 1.27980015],
                          [1.79888576, 0.92972442, 1.2054906, 1.32186075]])
    test_chiks_q = np.array([0.36507507, 0.19186653, 0.23056424, 0.24705505])
    test_mw_pq = np.array([[0., 0.66521177, 0.46738581, 0.3870413],
                           [0., 0.84222041, 0.57640002, 0.48426118],
                           [0., 4.05369028, 3.07212255, 3.05623433],
                           [0., 0.47398746, 0.32620549, 0.26836406],
                           [0., 0.78145439, 0.53931965, 0.45636238],
                           [0., 3.17848551, 2.39173871, 2.30227841],
                           [0., 0.78656857, 0.5370069, 0.43169684]])

    # Exchange constants
    assert np.allclose(J_qp.imag, 0.)
    assert np.allclose(J_qp.real, test_J_pq.T, rtol=1e-3)

    # Static reactive part of chiks
    assert np.allclose(chiks_GGq[0, 0, :].imag, 0.)
    assert np.allclose(chiks_GGq[0, 0, :].real, test_chiks_q, rtol=5.e-3)
    
    # Magnon energies
    assert np.allclose(mw_qp, test_mw_pq.T, rtol=1e-3)
