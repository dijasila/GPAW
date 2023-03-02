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


def generate_system_s():
    # Compute chiks for different materials and spin-components, using
    # system specific tolerances
    system_s = [  # wfs, spincomponent, rtol, dsym_rtol, bsum_rtol
        ('fe_pw_wfs', '+-', 0.04, 0.01, 0.02),
        ('fe_pw_wfs', '00', 0.04, 0.01, 0.02)
    ]

    return system_s


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
@pytest.mark.parametrize(
    'system,q_c,gammacentered',
    product(generate_system_s(), generate_q_qc(), generate_gc_g()))
def test_chiks_symmetry(in_tmp_dir, gpw_files, system, q_c, gammacentered):
    r"""Check the reciprocity relation (valid both for μν=00 and μν=+-),

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,-G'-G)^(μν)(-q, ω),

    the inversion symmetry relation,

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,-G-G')^(μν)(-q, ω),

    and the combination of the two,

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,G'G)^(μν)(q, ω),

    for a real life periodic systems with inversion symmetry.

    Unfortunately, there will always be random noise in the wave functions,
    such that these symmetries cannot be fulfilled exactly. Generally speaking,
    the "symmetry" noise can be reduced by running with symmetry='off' in
    the ground state calculation.

    Also, we test that the response function does not change too much when
    including symmetries in the response calculation and when changing between
    different band summation schemes."""

    # ---------- Inputs ---------- #

    # Part 1: ChiKS calculation
    wfs, spincomponent, rtol, dsym_rtol, bsum_rtol = system

    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    complex_frequencies = list(frequencies + 0.j) + list(frequencies + 0.1j)

    # Calculation parameters (which should not affect the result)
    disable_syms_s = [True, False]
    bandsummation_b = ['double', 'pairwise']
    bundle_integrals_i = [True, False]

    if world.size % 4 == 0:
        nblocks = 4
    elif world.size % 2 == 0:
        nblocks = 2
    else:
        nblocks = 1

    # Part 2: Check reciprocity and inversion symmetry

    # Part 3: Check toggling of calculation parameters
    bint_rtol = 1e-6

    # ---------- Script ---------- #

    # Part 1: ChiKS calculation
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']

    gs = ResponseGroundStateAdapter(calc)

    # Calculate chiks for q and -q
    if np.allclose(q_c, 0.):
        q_qc = [q_c]
    else:
        q_qc = [-q_c, q_c]

    # Set up complex frequency descriptor
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    chiks_sbiq = []
    for disable_syms in disable_syms_s:
        chiks_biq = []
        for bandsummation in bandsummation_b:
            chiks_iq = []
            for bundle_integrals in bundle_integrals_i:
                chiks_q = []

                chiks_calc = ChiKSCalculator(
                    gs, ecut=ecut, nbands=nbands,
                    gammacentered=gammacentered,
                    disable_time_reversal=disable_syms,
                    disable_point_group=disable_syms,
                    bandsummation=bandsummation,
                    bundle_integrals=bundle_integrals,
                    nblocks=nblocks)

                for q_c in q_qc:
                    chiks = chiks_calc.calculate(spincomponent, q_c, zd)
                    chiks = chiks.copy_with_global_frequency_distribution()
                    chiks_q.append(chiks)

                chiks_iq.append(chiks_q)
            chiks_biq.append(chiks_iq)
        chiks_sbiq.append(chiks_biq)

    # Part 2: Check reciprocity and inversion symmetry
    for chiks_biq in chiks_sbiq:
        for chiks_iq in chiks_biq:
            for chiks_q in chiks_iq:
                check_reciprocity_and_inversion_symmetry(chiks_q, rtol=rtol)

    # Part 3: Check toggling of calculation parameters

    # Check that all plane wave representations are identical
    sbi_s = list(product([0, 1], [0, 1], [0, 1]))
    for s, sbi1 in enumerate(sbi_s):
        for sbi2 in sbi_s[s + 1:]:
            for chiks1, chiks2 in zip(chiks_sbiq[sbi1[0]][sbi1[1]][sbi1[2]],
                                      chiks_sbiq[sbi2[0]][sbi2[1]][sbi2[2]]):
                qpd1, qpd2 = chiks1.qpd, chiks2.qpd
                G1_Gc = get_pw_coordinates(qpd1)
                G2_Gc = get_pw_coordinates(qpd2)
                assert G1_Gc.shape == G2_Gc.shape
                assert np.allclose(G1_Gc - G2_Gc, 0.)

    # Check symmetry toggle
    for b in range(2):
        for i in range(2):
            check_arrays(chiks_sbiq, (0, b, i), (1, b, i), rtol=dsym_rtol)

    # Check bandsummation toggle
    for s in range(2):
        for i in range(2):
            check_arrays(chiks_sbiq, (s, 0, i), (s, 1, i), rtol=bsum_rtol)

    # Check bundle_integrals toggle
    for s in range(2):
        for b in range(2):
            check_arrays(chiks_sbiq, (s, b, 0), (s, b, 1), rtol=bint_rtol)


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


def check_arrays(chiks_sbiq, sbi1, sbi2, *, rtol):
    """Compare the values inside two arrays."""
    s1, b1, i1 = sbi1
    s2, b2, i2 = sbi2
    chiks1_q = chiks_sbiq[s1][b1][i1]
    chiks2_q = chiks_sbiq[s2][b2][i2]
    for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
        assert chiks2.array == pytest.approx(chiks1.array, rel=rtol)
