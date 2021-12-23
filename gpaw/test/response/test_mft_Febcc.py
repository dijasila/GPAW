# Import modules
from gpaw import GPAW, PW, FermiDirac
from gpaw.test import equal
from ase.build import bulk
import numpy as np
from gpaw.response.mft import IsotropicExchangeCalculator, \
    compute_magnon_energy_simple, compute_magnon_energy_FM


def test_Fe_bcc():
    a = 2.867
    mm = 2.21
    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])

    # Calculation settings
    k = 4
    pw = 200
    xc = 'LDA'
    nbands_gs = 10   # Number of bands in ground state calculation
    # Number of bands to converge and use for response calculation
    nbands_response = 8
    conv = {'density': 1e-8,
            'forces': 1e-8,
            'bands': nbands_response}
    ecut = 50
    sitePos_mv = atoms.positions
    shapes_m = 'sphere'
    # All high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     [0.25, 0.25, 0.25]   # P
                     ])

    # Construct calculator
    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (k, k, k), 'gamma': True},
                nbands=nbands_gs,
                occupations=FermiDirac(0.01),
                symmetry={'point_group': False},
                idiotproof=False,
                parallel={'domain': 1},
                spinpol=True,
                convergence=conv
                )

    # Do ground state calculation
    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    # Prepare exchange calculator
    exchCalc = IsotropicExchangeCalculator(calc,
                                           sitePos_mv,
                                           shapes_m=shapes_m,
                                           ecut=ecut,
                                           nbands=nbands_response)

    # Do calculation for each q-point
    J_rmnq = np.zeros([1, 1, 1, 4], dtype=np.complex128)
    for q, q_c in enumerate(q_qc):
        J_rmnq[:, :, :, q] = exchCalc(q_c, rc_rm=1)
    J_q = J_rmnq[0, 0, 0, :]

    # Compare with expected result (previous calculation with working code)
    Jexp_q = np.array([0.06138644+0.0j, 0.03409565+0.0j,
                       0.04244621+0.0j, 0.04533697+0.0j])
    equal(J_q, Jexp_q, 1e-5)

    # Test Bxc and chiks calculators individually
    Bxc_G = exchCalc.Bxc_G
    chiks_GGq = []
    for q_c in q_qc:
        _, chiks_GG = exchCalc.chiksf('-+', q_c, txt=None)
        chiks_GGq += [chiks_GG]
    chiks_GGq = np.dstack(chiks_GGq)
    chiksexp_q = np.array([0.36527329, 0.21086173, 0.2479018, 0.26542496])
    equal(chiksexp_q, chiks_GGq[0, 0, :], 1e-4)
    Bxcexp_G = np.array([-0.82801687, -0.28927704, -0.28927704, -0.28927704])
    equal(Bxcexp_G, Bxc_G[:4], 1e-4)

    # Test computation of magnon energies
    Efm_mq = compute_magnon_energy_FM(J_rmnq[0, :, :, :], q_qc, mm)
    Esimple_q = compute_magnon_energy_simple(J_q, q_qc, mm)
    equal(Efm_mq[0, :], Esimple_q, 1e-10)
    Eexp_q = np.array([0., 0.02469755, 0.01714048, 0.01452441])
    equal(Eexp_q, Efm_mq[0, :], 1e-5)
