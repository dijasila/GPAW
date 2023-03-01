"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

from itertools import product

import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import world
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


def generate_gc_g():
    # Compute chiks both on a gamma-centered and a q-centered pw grid
    gc_g = [True, False]

    return gc_g


# ---------- Actual tests ---------- #


@pytest.mark.response
@pytest.mark.parametrize('q_c,gammacentered', product(generate_q_qc(),
                                                      generate_gc_g()))
def test_transverse_chiks_symmetry(in_tmp_dir, gpw_files,
                                   q_c, gammacentered):
    """Check the reciprocity relation,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G'-G)^(+-)(-q, ω),

    the inversion symmetry relation,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G-G')^(+-)(-q, ω),

    and the combination of the two,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,G'G)^(+-)(q, ω),

    for a real life system with d-electrons and inversion symmetry (bcc-Fe).

    Unfortunately, there will always be random noise in the wave functions,
    such that these symmetries are not fulfilled exactly. However, we should be
    able to fulfill it within 4%, which is tested here. Generally speaking,
    the "symmetry" noise can be reduced making running with symmetry='off' in
    the ground state calculation.

    Also, we test that the response function does not change by toggling
    the symmetries within the same precision."""

    # ---------- Inputs ---------- #

    # Part 1: ChiKS calculation
    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    complex_frequencies = list(frequencies + 0.j) + list(frequencies + 0.1j)

    # Calculation parameters (which should not affect the result)
    disable_syms_s = [True, False]
    bandsummation_b = ['double', 'pairwise']

    if world.size % 4 == 0:
        nblocks = 4
    elif world.size % 2 == 0:
        nblocks = 2
    else:
        nblocks = 1

    # Part 2: Check reciprocity and inversion symmetry
    rtol = 0.04

    # Part 3: Check toggling of calculation parameters
    dsym_rtol = 0.01
    bsum_rtol = 0.02

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
                                         bandsummation=bandsummation,
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
            check_reciprocity_and_inversion_symmetry(chiks_q, rtol=rtol)

    # Part 3: Check toggling of calculation parameters

    # Check that all plane wave representations are identical
    sb_s = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for s, sb1 in enumerate(sb_s):
        for sb2 in sb_s[s + 1:]:
            for chiks1, chiks2 in zip(chiks_sbq[sb1[0]][sb1[1]],
                                      chiks_sbq[sb2[0]][sb2[1]]):
                qpd1, qpd2 = chiks1.qpd, chiks2.qpd
                G1_Gc = get_pw_coordinates(qpd1)
                G2_Gc = get_pw_coordinates(qpd2)
                assert G1_Gc.shape == G2_Gc.shape
                assert np.allclose(G1_Gc - G2_Gc, 0.)

    # Check symmetry toggle
    for b in range(2):
        chiks1_q = chiks_sbq[0][b]
        chiks2_q = chiks_sbq[1][b]
        for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
            assert chiks2.array == pytest.approx(chiks1.array, rel=dsym_rtol)

    # Check bandsummation toggle
    for s in range(2):
        chiks1_q = chiks_sbq[s][0]
        chiks2_q = chiks_sbq[s][1]
        for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
            assert chiks2.array == pytest.approx(chiks1.array, rel=bsum_rtol)


# ---------- Test functionality ---------- #


def check_reciprocity_and_inversion_symmetry(chiks_q, *, rtol):
    """Carry out the actual susceptibility symmetry checks."""
    # Get the q and -q pair
    if len(chiks_q) == 2:
        q1, q2 = 0, 1
    else:
        assert len(chiks_q) == 1
        assert np.allclose(chiks_q[0].q_c, 0.)
        q1, q2 = 0, 0

    qpd1, qpd2 = chiks_q[q1].qpd, chiks_q[q2].qpd
    invmap_GG = get_inverted_pw_mapping(qpd1, qpd2)

    # Loop over frequencies
    for chi1_GG, chi2_GG in zip(chiks_q[q1].array,
                                chiks_q[q2].array):
        # Check the reciprocity
        assert chi2_GG[invmap_GG].T == pytest.approx(chi1_GG, rel=rtol)
        # Check inversion symmetry
        assert chi2_GG[invmap_GG] == pytest.approx(chi1_GG, rel=rtol)

    # Loop over q-vectors
    for chiks in chiks_q:
        for chiks_GG in chiks.array:  # array = chiks_zGG
            # Check that the full susceptibility matrix is symmetric
            assert chiks_GG.T == pytest.approx(chiks_GG, rel=rtol)
