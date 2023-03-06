"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

from itertools import product, combinations

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
        ('fancy_si_pw_wfs', '00', 1e-5, 1e-6, 1e-5),
        ('al_pw_wfs', '00', 1e-5, 4.0, 1e-5),  # unstable symmetry -> #788
        ('fe_pw_wfs', '00', 1e-5, 1e-6, 1e-5),
        ('fe_pw_wfs', '+-', 0.04, 0.01, 0.02)
    ]

    return system_s


def generate_qrel_q():
    # Fractional q-vectors on a path towards a reciprocal lattice vector
    qrel_q = np.array([0., 0.25, 0.5])

    return qrel_q


def get_q_c(wfs, qrel):
    if wfs in ['fancy_si_pw_wfs', 'al_pw_wfs']:
        # Generate points on the G-X path
        q_c = qrel * np.array([1., 0., 1.])
    elif wfs == 'fe_pw_wfs':
        # Generate points on the G-N path
        q_c = qrel * np.array([0., 0., 1.])
    else:
        raise ValueError('Invalid wfs', wfs)

    return q_c


def generate_gc_g():
    # Compute chiks both on a gamma-centered and a q-centered pw grid
    gc_g = [True, False]

    return gc_g


# ---------- Actual tests ---------- #


def mark_si_xfail(system, request):
    # Bug tracked in #802
    wfs = system[0]
    if wfs == 'fancy_si_pw_wfs' and world.size % 4 == 0:
        request.node.add_marker(pytest.mark.xfail)


@pytest.mark.response
@pytest.mark.parametrize(
    'system,qrel,gammacentered',
    product(generate_system_s(), generate_qrel_q(), generate_gc_g()))
def test_chiks_symmetry(in_tmp_dir, gpw_files, system, qrel, gammacentered,
                        request):
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

    Also, we test that the susceptibility does not change due to varrying block
    distribution, band summation scheme or when reducing the k-point integral
    using symmetries."""
    mark_si_xfail(system, request)

    # ---------- Inputs ---------- #

    # Part 1: ChiKS calculation
    wfs, spincomponent, rtol, dsym_rtol, bsum_rtol = system
    q_c = get_q_c(wfs, qrel)

    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    complex_frequencies = list(frequencies + 0.j) + list(frequencies + 0.1j)

    # Calculation parameters (which should not affect the result)
    nblocks_n = [1]
    if world.size % 2 == 0:
        nblocks_n.append(2)
    if world.size % 4 == 0:
        nblocks_n.append(4)
    nn = len(nblocks_n)

    disable_syms_s = [True, False]
    bandsummation_b = ['double', 'pairwise']
    bundle_integrals_i = [True, False]

    # Part 2: Check toggling of calculation parameters
    nblocks_rtol = 1e-6
    bint_rtol = 1e-6

    # Part 3: Check reciprocity and inversion symmetry

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

    chiks_nsbiq = []
    for nblocks in nblocks_n:
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
        chiks_nsbiq.append(chiks_sbiq)

    # Part 2: Check toggling of calculation parameters

    # Check that all plane wave representations are identical
    nsbi_s = list(product(range(nn), [0, 1], [0, 1], [0, 1]))
    for s, nsbi1 in enumerate(nsbi_s):
        for nsbi2 in nsbi_s[s + 1:]:
            chiks1_q = chiks_nsbiq[nsbi1[0]][nsbi1[1]][nsbi1[2]][nsbi1[3]]
            chiks2_q = chiks_nsbiq[nsbi2[0]][nsbi2[1]][nsbi2[2]][nsbi2[3]]
            for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
                qpd1, qpd2 = chiks1.qpd, chiks2.qpd
                G1_Gc = get_pw_coordinates(qpd1)
                G2_Gc = get_pw_coordinates(qpd2)
                assert G1_Gc.shape == G2_Gc.shape
                assert np.allclose(G1_Gc - G2_Gc, 0.)

    # Check nblocks toggle
    for s in range(2):
        for b in range(2):
            for i in range(2):
                for n1, n2 in combinations(range(nn), 2):
                    check_arrays(chiks_nsbiq, (n1, s, b, i), (n2, s, b, i),
                                 rtol=nblocks_rtol)

    # Check symmetry toggle
    for n in range(nn):
        for b in range(2):
            for i in range(2):
                check_arrays(chiks_nsbiq, (n, 0, b, i), (n, 1, b, i),
                             rtol=dsym_rtol)

    # Check bandsummation toggle
    for n in range(nn):
        for s in range(2):
            for i in range(2):
                check_arrays(chiks_nsbiq, (n, s, 0, i), (n, s, 1, i),
                             rtol=bsum_rtol)

    # Check bundle_integrals toggle
    for n in range(nn):
        for s in range(2):
            for b in range(2):
                check_arrays(chiks_nsbiq, (n, s, b, 0), (n, s, b, 1),
                             rtol=bint_rtol)

    # Part 3: Check reciprocity and inversion symmetry
    for chiks_sbiq in chiks_nsbiq:
        for chiks_biq in chiks_sbiq:
            for chiks_iq in chiks_biq:
                for chiks_q in chiks_iq:
                    check_reciprocity_and_inversion_symmetry(
                        chiks_q, rtol=rtol)


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

    qpd1 = chiks_q[q1].qpd
    qpd2 = chiks_q[q2].qpd
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


def check_arrays(chiks_nsbiq, nsbi1, nsbi2, *, rtol):
    """Compare the values inside two arrays."""
    n1, s1, b1, i1 = nsbi1
    n2, s2, b2, i2 = nsbi2
    chiks1_q = chiks_nsbiq[n1][s1][b1][i1]
    chiks2_q = chiks_nsbiq[n2][s2][b2][i2]
    for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
        assert chiks2.array == pytest.approx(chiks1.array, rel=rtol, abs=1e-8)
