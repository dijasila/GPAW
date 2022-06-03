"""Calculate the Heisenberg exchange constants in Fe using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""
from gpaw import GPAW, PW, FermiDirac
from ase.build import bulk
import numpy as np
from gpaw.response.mft import IsotropicExchangeCalculator
from gpaw.response.heisenberg import calculate_FM_magnon_energies


def test_Fe_bcc():
    # ---------- Inputs ---------- #

    # Part 1: ground state calculation
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
    shapes_m = 'sphere'  # Test all the site topologies? XXX
    rc_rm = np.array([[1.0], [1.5], [2.0]])

    # ---------- Script ---------- #

    # Part 1: ground state calculation

    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])

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

    sitePos_mv = atoms.positions  # Using Fe atom as the site
    exchCalc = IsotropicExchangeCalculator(calc,
                                           sitePos_mv,
                                           shapes_m=shapes_m,
                                           ecut=ecut,
                                           nbands=nbands)

    # Calcualate the exchange constant for each q-point
    J_qabr = np.empty((len(q_qc), 1, 1, len(rc_rm)), dtype=complex)
    for q, q_c in enumerate(q_qc):
        J_qabr[q] = exchCalc(q_c, rc_rm=rc_rm)
    J_qr = J_qabr[:, 0, 0, :]

    # Calculate the magnon energies
    mw_qnr = calculate_FM_magnon_energies(J_qabr, q_qc,
                                          mm * np.ones((1, len(rc_rm))))

    # Run the chiks calculator individually
    chiks_GGq = []  # Could be tested elsewhere? XXX
    for q_c in q_qc:
        _, chiks_GG = exchCalc.chiksf('-+', q_c, txt=None)
        chiks_GGq += [chiks_GG]
    chiks_GGq = np.dstack(chiks_GGq)

    # Part 3: compare new results to test values
    test_J_rq = np.array([[1.61643955, 0.88155322, 1.10019274, 1.18879169],
                          [1.8678718, 0.93756859, 1.23108965, 1.33281237],
                          [4.67944783, 0.20054973, 1.28535702, 1.30257353]])
    test_chiks_q = np.array([0.36507507, 0.19186653, 0.23056424, 0.24705505])
    test_mw_rq = np.array([[0., 0.6650555, 0.46719168, 0.38701164],
                           [0., 0.84204307, 0.57626353, 0.48421668],
                           [0., 4.05334907, 3.07152699, 3.05599481]])

    # Exchange constants
    assert np.allclose(J_qr.imag, 0.)
    assert np.allclose(J_qr.real, test_J_rq.T, rtol=1e-3)

    # Static reactive part of chiks
    assert np.allclose(chiks_GGq[0, 0, :].imag, 0.)
    assert np.allclose(chiks_GGq[0, 0, :].real, test_chiks_q, rtol=1e-3)
    
    # Magnon energies
    assert np.allclose(mw_qnr[:, 0, :], test_mw_rq.T, rtol=1e-3)
