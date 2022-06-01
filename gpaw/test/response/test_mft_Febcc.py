# Import modules
from gpaw import GPAW, PW, FermiDirac
from gpaw.test import equal
from ase.build import bulk
import numpy as np
from gpaw.response.mft import IsotropicExchangeCalculator, \
    compute_magnon_energy_simple, compute_magnon_energy_FM


def test_Fe_bcc():
    # ---------- Inputs ---------- #

    # Part 1: ground state calculation
    xc = 'LDA'
    kpts = 4
    nbands = 8
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
    shapes_m = 'sphere'
    rc_rm = np.array([[1.0], [1.5], [2.0]])

    # ---------- Script ---------- #

    # Part 1: ground state calculation

    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                nbands=nbands + 2,
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
    J_rmnq = np.zeros([len(rc_rm), 1, 1, len(q_qc)], dtype=np.complex128)
    for q, q_c in enumerate(q_qc):
        J_rmnq[:, :, :, q] = exchCalc(q_c, rc_rm=rc_rm)
    J_q = J_rmnq[0, 0, 0, :]

    # Calculate the magnon energies
    mw_mq = compute_magnon_energy_FM(J_rmnq[0, :, :, :], q_qc, mm)
    mw_q = compute_magnon_energy_simple(J_q, q_qc, mm)

    # Run the chiks calculator individually
    chiks_GGq = []  # Could be tested elsewhere? XXX
    for q_c in q_qc:
        _, chiks_GG = exchCalc.chiksf('-+', q_c, txt=None)
        chiks_GGq += [chiks_GG]
    chiks_GGq = np.dstack(chiks_GGq)

    # Part 3: compare new results to test values
    test_J_q = np.array([1.67041012 + 0.0j, 0.92778989 + 0.0j,
                         1.15502021 + 0.0j, 1.23368179 + 0.0j])
    test_chiks_q = np.array([0.36527329, 0.21086173, 0.2479018, 0.26542496])
    test_Bxc_G = np.array([-0.82801687, -0.28927704, -0.28927704, -0.28927704])
    test_mw_q = np.array([0., 0.67205457, 0.46641622, 0.39522933])

    # Exchange constants
    equal(J_q, test_J_q, 1e-3)

    # Bxc field
    Bxc_G = exchCalc.Bxc_G  # Could be tested elsewhere? XXX
    equal(Bxc_G[:4], test_Bxc_G, 1e-4)

    # Static reactive part of chiks
    equal(chiks_GGq[0, 0, :], test_chiks_q, 1e-4)
    
    # Magnon energies
    equal(mw_mq[0, :], mw_q, 1e-10)  # Check for self-consistency
    equal(mw_mq[0, :], test_mw_q, 1e-4)
