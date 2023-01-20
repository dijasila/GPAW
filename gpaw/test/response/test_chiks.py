"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

from itertools import product

import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import rank, world
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import (get_inverted_pw_mapping,
                                          get_pw_coordinates)

# ---------- ChiKS parametrization ---------- #


def generate_q_qc():
    # Do some q-points on the G-N path
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0., 0., 0.25],      # N/2
                     [0.0, 0.0, 0.5],     # N
                     ])

    return q_qc


def generate_eta_e():
    # Try out both a vanishing and finite broadening
    eta_e = [0., 0.1]

    return eta_e


def generate_gc_g():
    # Compute chiks both on a gamma-centered and a q-centered pw grid
    gc_g = [True, False]

    return gc_g


# ---------- Actual tests ---------- #


@pytest.mark.response
@pytest.mark.parametrize('q_c,eta,gammacentered', product(generate_q_qc(),
                                                          generate_eta_e(),
                                                          generate_gc_g()))
def test_chiks_symmetry(in_tmp_dir, gpw_files, q_c, eta, gammacentered):
    """Check the reciprocity relation,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G'-G)^(+-)(-q, ω),

    the inversion symmetry relation,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G-G')^(+-)(-q, ω),

    and the combination of the two,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,G'G)^(+-)(q, ω),

    for a real life system with d-electrons and inversion symmetry (bcc-Fe).

    Unfortunately, there will always be random noise in the wave functions,
    such that these symmetries are not fulfilled exactly. However, we should be
    able to fulfill it within 3%, which is tested here. Generally speaking,
    the "symmetry" noise can be reduced making running with symmetry='off' in
    the ground state calculation.

    Also, we test that the response function does not change by toggling
    the symmetries within the same precision."""

    # ---------- Inputs ---------- #

    # Part 1: ChiKS calculation
    ecut = 50
    frequencies = [0., 0.05, 0.1, 0.2]
    disable_syms_s = [True, False]
    bandsummation_b = ['double', 'pairwise']

    if world.size % 4 == 0:
        nblocks = 4
    elif world.size % 2 == 0:
        nblocks = 2
    else:
        nblocks = 1

    # Part 2: Check reciprocity and inversion symmetry
    rtol = 0.035

    # Part 3: Check matrix symmetry

    # Part 4: Check symmetry and bandsummation toggles
    trtol = 0.007

    # ---------- Script ---------- #

    # Part 1: ChiKS calculation
    calc = GPAW(gpw_files['fe_pw_wfs'], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']

    gs = ResponseGroundStateAdapter(calc)

    # Calculate chiks for q and -q
    if np.allclose(q_c, 0.):
        q_qc = [q_c]
    else:
        q_qc = [-q_c, q_c]

    # Set up complex frequency descriptor
    complex_frequencies = np.array(frequencies) + 1.j * eta
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    chiks_sbq = []
    for disable_syms in disable_syms_s:
        chiks_bq = []
        for bandsummation in bandsummation_b:
            chiks_calc = ChiKSCalculator(gs,
                                         ecut=ecut, nbands=nbands,
                                         gammacentered=gammacentered,
                                         disable_time_reversal=disable_syms,
                                         disable_point_group=disable_syms,
                                         nblocks=nblocks)

            chiks_q = []
            for q_c in q_qc:
                chiks = chiks_calc.calculate('+-', q_c, zd)
                chiks = chiks.copy_with_global_frequency_distribution()
                chiks_q.append(chiks)

            chiks_bq.append(chiks_q)
        chiks_sbq.append(chiks_bq)

    # Part 2: Check reciprocity and inversion symmetry
    for chiks_bq in chiks_sbq:
        for chiks_q in chiks_bq:
            # Get the q and -q pair
            if len(chiks_q) == 2:
                q1, q2 = 0, 1
            else:
                assert len(chiks_q) == 1
                assert np.allclose(q_c, 0.)
                q1, q2 = 0, 0

            pd1, pd2 = chiks_q[q1].pd, chiks_q[q2].pd
            invmap_GG = get_inverted_pw_mapping(pd1, pd2)

            # Check reciprocity of the reactive part of the static
            # susceptibility. This specific check makes sure that the
            # exchange constants calculated within the MFT remains
            # reciprocal.
            if rank == 0:  # Only the root has the static susc.
                # Calculate the reactive part
                chi1_GG = chiks_q[q1].array[0]  # array = chiks_wGG
                chi2_GG = chiks_q[q2].array[0]
                chi1r_GG = 1 / 2. * (chi1_GG + np.conj(chi1_GG).T)
                chi2r_GG = 1 / 2. * (chi2_GG + np.conj(chi2_GG).T)
                assert np.conj(chi2r_GG[invmap_GG]) == pytest.approx(chi1r_GG,
                                                                     rel=rtol)

            # Loop over frequencies
            for chi1_GG, chi2_GG in zip(chiks_q[q1].array,
                                        chiks_q[q2].array):
                # Check the reciprocity of the full susceptibility
                assert chi2_GG[invmap_GG].T == pytest.approx(chi1_GG, rel=rtol)
                # Check inversion symmetry of the full susceptibility
                assert chi2_GG[invmap_GG] == pytest.approx(chi1_GG, rel=rtol)

    # Part 3: Check matrix symmetry
    for chiks_bq in chiks_sbq:
        for chiks_q in chiks_bq:
            for chiks in chiks_q:
                for chiks_GG in chiks.array:  # array = chiks_wGG
                    assert chiks_GG.T == pytest.approx(chiks_GG, rel=rtol)

    # Part 4: Check symmetry and bandsummation toggles

    # Check that the plane wave representations are identical
    sb_s = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for s, sb1 in enumerate(sb_s):
        for sb2 in sb_s[s + 1:]:
            for chiks1, chiks2 in zip(chiks_sbq[sb1[0]][sb1[1]],
                                      chiks_sbq[sb2[0]][sb2[1]]):
                pd1, pd2 = chiks1.pd, chiks2.pd
                G1_Gc = get_pw_coordinates(pd1)
                G2_Gc = get_pw_coordinates(pd2)
                assert G1_Gc.shape == G2_Gc.shape
                assert np.allclose(G1_Gc - G2_Gc, 0.)

    for s, sb1 in enumerate(sb_s):
        for sb2 in sb_s[s + 1:]:
            chiks1_q = chiks_sbq[sb1[0]][sb1[1]]
            chiks2_q = chiks_sbq[sb2[0]][sb2[1]]
            for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
                assert chiks2.array == pytest.approx(chiks1.array,
                                                     rel=trtol)
