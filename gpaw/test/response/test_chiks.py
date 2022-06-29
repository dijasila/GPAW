"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

# General modules
import pytest
import numpy as np

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.chiks import ChiKS
from gpaw.response.susceptibility import get_pw_coordinates


# ---------- Actual tests ---------- #


@pytest.mark.response
def test_Fe_chiks(in_tmp_dir):
    """Check the reciprocity relation

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G'-G)^(+-)(-q, ω)

    for a real life system with d-electrons (bcc-Fe)."""
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

    # Part 2: ChiKS calculation
    ecut = 50
    eta_e = [0., 0.1]
    frequencies = [0., 0.05, 0.1, 0.2]
    # Do some q-points on the -N..G..N path
    q_qc = np.array([[0., 0., -0.5],      # -N
                     [0., 0., -0.25],     # -N/2
                     [0, 0, 0],           # Gamma
                     [0., 0., 0.25],      # N/2
                     [0.0, 0.0, 0.5],     # N
                     ])

    # Part 3: Check reciprocity

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
                idiotproof=False,
                parallel={'domain': 1},
                spinpol=True,
                convergence=conv
                )

    atoms.calc = calc
    atoms.get_potential_energy()

    # Part 2: ChiKS calculation
    chiks_eqwGG = []
    pd_eq = []
    for eta in eta_e:
        chiks = ChiKS(calc,
                      ecut=ecut, nbands=nbands, eta=eta,
                      disable_time_reversal=False,  # Make this the default XXX
                      disable_point_group=False)

        chiks_qwGG = []
        pd_q = []
        for q_c in q_qc:
            pd, chiks_wGG = chiks.calculate(q_c, frequencies,
                                            spincomponent='+-')
            pd_q.append(pd)
            chiks_qwGG.append(chiks_wGG)

        chiks_eqwGG.append(chiks_qwGG)
        pd_eq.append(pd_q)

    # Part 3: Check reciprocity

    for pd_q, chiks_qwGG in zip(pd_eq, chiks_eqwGG):  # One eta at a time
        for q1, q2 in zip([0, 1, 2], [4, 3, 2]):  # q and -q pairs
            invmap_GG = get_inverted_pw_mapping(pd_q[q1], pd_q[q2])

            # Check reciprocity of the reactive part of the static
            # susceptibility. This specific check makes sure that the exchange
            # constants calculated within the MFT remains reciprocal.
            # Calculate the reactive part
            chi1_GG, chi2_GG = chiks_qwGG[q1][0], chiks_qwGG[q2][0]
            chi1r_GG = 1 / 2. * (chi1_GG + np.conj(chi1_GG).T)
            chi2r_GG = 1 / 2. * (chi2_GG + np.conj(chi2_GG).T)

            assert np.allclose(np.conj(chi1r_GG), chi2r_GG[invmap_GG])

            # Check the reciprocity of the full susceptibility
            assert np.allclose(chiks_qwGG[q1],
                               np.transpose(chiks_qwGG[q2][:, invmap_GG],
                                            (0, 2, 1)))


# ---------- Test functionality ---------- #


def get_inverted_pw_mapping(pd1, pd2):
    """Get the plane wave coefficients mapping GG' of pd1 into -G-G' of pd2"""
    G1_Gc = get_pw_coordinates(pd1)
    G2_Gc = get_pw_coordinates(pd2)

    mG2_G1 = []
    for G1_c in G1_Gc:
        found_match = False
        for G2, G2_c in enumerate(G2_Gc):
            if np.all(G2_c == -G1_c):
                mG2_G1.append(G2)
                found_match = True
                break
        if not found_match:
            raise ValueError('Could not match pd1 and pd2')

    # Set up mapping from GG' to -G-G'
    invmap_GG = np.meshgrid(mG2_G1, mG2_G1, indexing='ij')

    return invmap_GG
