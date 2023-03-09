"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

from itertools import product, combinations

import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.chi0 import Chi0
from gpaw.response.susceptibility import (get_inverted_pw_mapping,
                                          get_pw_coordinates)

# ---------- ChiKS parametrization ---------- #


def generate_system_s(spincomponents=['00', '+-']):
    # Compute chiks for different materials and spin-components, using
    # system specific tolerances
    system_s = [  # wfs, spincomponent, rtol, dsym_rtol, bsum_rtol
        ('fancy_si_pw_wfs', '00', 1e-5, 1e-6, 1e-5),
        ('al_pw_wfs', '00', 1e-5, 4.0, 1e-5),  # unstable symmetry -> #788
        ('fe_pw_wfs', '00', 1e-5, 1e-6, 1e-5),
        ('fe_pw_wfs', '+-', 0.04, 0.01, 0.02)
    ]

    # Filter spincomponents
    system_s = [system for system in system_s if system[1] in spincomponents]

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
def test_chiks(in_tmp_dir, gpw_files, system, qrel, gammacentered, request):
    r"""Test the internals of the ChiKSCalculator.

    In particular, we test that the susceptibility does not change due to the
    details in the internal calculator, such as varrying block distribution,
    band summation scheme, reducing the k-point integral using symmetries or
    basing the ground state adapter on a dynamic (and distributed) GPAW
    calculator.

    Furthermore, we test the symmetries of the calculated susceptibilities.
    In particular, we test the reciprocity relation (valid both for μν=00 and
    μν=+-),

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,-G'-G)^(μν)(-q, ω),

    the inversion symmetry relation,

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,-G-G')^(μν)(-q, ω),

    and the combination of the two,

    χ_(KS,GG')^(μν)(q, ω) = χ_(KS,G'G)^(μν)(q, ω),

    for a real life periodic systems with an inversion center.

    Unfortunately, there will always be random noise in the wave functions,
    such that these symmetries cannot be fulfilled exactly. Generally speaking,
    the "symmetry" noise can be reduced by running with symmetry='off' in
    the ground state calculation."""
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
    dynamic_ground_state_d = [True, False]
    disable_syms_s = [True, False]

    nblocks_n = [1]
    if world.size % 2 == 0:
        nblocks_n.append(2)
    if world.size % 4 == 0:
        nblocks_n.append(4)
    nn = len(nblocks_n)

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

    context = ResponseContext()

    # Calculate chiks for q and -q
    if np.allclose(q_c, 0.):
        q_qc = [q_c]
    else:
        q_qc = [-q_c, q_c]

    # Set up complex frequency descriptor
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    # Run calculations to cross-validate dynamic_ground_state,
    # disable_symmetries, nblocks and bandsummation
    chiks_dsnbq = []
    for dynamic_ground_state in dynamic_ground_state_d:
        chiks_snbq = []
        gs = initialize_ground_state_adapter(
            gpw_files[wfs], calc, context,
            dynamic_ground_state=dynamic_ground_state)
        for disable_syms in disable_syms_s:
            chiks_nbq = []
            if gammacentered and not np.allclose(q_c, 0.):
                # When calculating chiks on a gammacentered grid, the
                # plane-wave representation used internally depends on
                # whether symmetry is used in the calculation or not. This
                # means that we need to reinitialize the ground state adapter
                # with a fresh PAW corrections cache.
                gs = initialize_ground_state_adapter(
                    gpw_files[wfs], calc, context,
                    dynamic_ground_state=dynamic_ground_state)
            for nblocks in nblocks_n:
                chiks_bq = []
                for bandsummation in bandsummation_b:
                    chiks_q = []

                    chiks_calc = ChiKSCalculator(
                        gs, context=context,
                        ecut=ecut, nbands=nbands,
                        gammacentered=gammacentered,
                        disable_time_reversal=disable_syms,
                        disable_point_group=disable_syms,
                        bandsummation=bandsummation,
                        nblocks=nblocks)

                    for q_c in q_qc:
                        chiks = chiks_calc.calculate(spincomponent, q_c, zd)
                        chiks = chiks.copy_with_global_frequency_distribution()
                        chiks_q.append(chiks)

                    chiks_bq.append(chiks_q)
                chiks_nbq.append(chiks_bq)
            chiks_snbq.append(chiks_nbq)
        chiks_dsnbq.append(chiks_snbq)

    # Run calculations to cross-validate nblocks and bundle_integrals
    d = -1  # Reuse the most recent ground state adapter with cached PAW
    s = -1  # corrections (i.e. also the most recent symmetry setting)
    b = 1  # Use the default pairwise band and spin summation scheme
    chiks_niq = []
    for n, nblocks in enumerate(nblocks_n):
        chiks_iq = []
        for bundle_integrals in bundle_integrals_i:
            if bundle_integrals:
                # This is the default setting, so we pick out the result from
                # the calculations above
                chiks_q = chiks_dsnbq[d][s][n][b]
            else:
                chiks_q = []

                chiks_calc = ChiKSCalculator(
                    gs, context=context,
                    ecut=ecut, nbands=nbands,
                    gammacentered=gammacentered,
                    bundle_integrals=bundle_integrals,
                    disable_time_reversal=disable_syms_s[s],
                    disable_point_group=disable_syms_s[s],
                    bandsummation=bandsummation_b[b],
                    nblocks=nblocks)

                for q_c in q_qc:
                    chiks = chiks_calc.calculate(spincomponent, q_c, zd)
                    chiks = chiks.copy_with_global_frequency_distribution()
                    chiks_q.append(chiks)

            chiks_iq.append(chiks_q)
        chiks_niq.append(chiks_iq)

    # Part 2: Check toggling of calculation parameters

    # Test that all plane-wave representations are identical
    dsnb_p = list(product(range(2), range(2), range(nn), range(2)))
    for p, dsnb1 in enumerate(dsnb_p):
        for dsnb2 in dsnb_p[p + 1:]:
            compare_dsnb_pw_bases(chiks_dsnbq, dsnb1, dsnb2)

    # Check dynamic ground state toggle
    for s in range(2):
        for n in range(nn):
            for b in range(2):
                compare_dsnb_arrays(chiks_dsnbq,
                                    (0, s, n, b), (1, s, n, b),
                                    rtol=dsym_rtol)

    # Check symmetry toggle
    for d in range(2):
        for n in range(nn):
            for b in range(2):
                compare_dsnb_arrays(chiks_dsnbq,
                                    (d, 0, n, b), (d, 1, n, b),
                                    rtol=dsym_rtol)

    # Check nblocks toggle
    for d in range(2):
        for s in range(2):
            for b in range(2):
                for n1, n2 in combinations(range(nn), 2):
                    compare_dsnb_arrays(chiks_dsnbq,
                                        (d, s, n1, b), (d, s, n2, b),
                                        rtol=nblocks_rtol)

    # Check bandsummation toggle
    for d in range(2):
        for s in range(2):
            for n in range(nn):
                compare_dsnb_arrays(chiks_dsnbq,
                                    (d, s, n, 0), (d, s, n, 1),
                                    rtol=bsum_rtol)

    # Check bundle_integrals toggle
    for chiks1_q, chiks2_q in chiks_niq:
        compare_pw_bases(chiks1_q, chiks2_q)
        compare_arrays(chiks1_q, chiks2_q, rtol=bint_rtol)

    # Part 3: Check reciprocity and inversion symmetry
    for chiks_snbq in chiks_dsnbq:
        for chiks_nbq in chiks_snbq:
            for chiks_biq in chiks_nbq:
                for chiks_q in chiks_bq:
                    check_reciprocity_and_inversion_symmetry(
                        chiks_q, rtol=rtol)
    for chiks_iq in chiks_niq:
        for chiks_q in chiks_iq:
            check_reciprocity_and_inversion_symmetry(chiks_q, rtol=rtol)


@pytest.mark.response
@pytest.mark.parametrize(
    'system,qrel',
    product(generate_system_s(spincomponents=['00']), generate_qrel_q()))
def test_chiks_vs_chi0(in_tmp_dir, gpw_files, system, qrel):
    """Test that the ChiKSCalculator is able to reproduce the Chi0Body.

    We use only the default calculation parameter setup for the ChiKSCalculator
    and leave parameter cross-validation to the test above."""

    # ---------- Inputs ---------- #

    # Part 1: ChiKS calculation
    wfs, spincomponent, rtol, _, _ = system
    q_c = get_q_c(wfs, qrel)

    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    eta = 0.15
    complex_frequencies = frequencies + 1.j * eta

    # Part 2: Chi0 calculation

    # Part 3: Check ChiKS vs. Chi0

    # ---------- Script ---------- #

    # Part 1: ChiKS calculation

    # Initialize ground state adapter
    context = ResponseContext()
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[wfs], context)
    nbands = gs._calc.parameters.convergence['bands']

    # Set up complex frequency descriptor
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    # Calculate ChiKS
    chiks_calc = ChiKSCalculator(gs, context=context,
                                 ecut=ecut, nbands=nbands)
    chiks = chiks_calc.calculate(spincomponent, q_c, zd)
    chiks = chiks.copy_with_global_frequency_distribution()

    # Part 2: Chi0 calculation
    chi0_calc = Chi0(gpw_files[wfs],
                     frequencies=frequencies, eta=eta,
                     ecut=ecut, nbands=nbands,
                     hilbert=False, intraband=False)
    chi0_data = chi0_calc.calculate(q_c)
    chi0_wGG = chi0_data.get_distributed_frequencies_array()

    # Part 3: Check ChiKS vs. Chi0
    assert chiks.array == pytest.approx(chi0_wGG, rel=rtol, abs=1e-8)


# ---------- Test functionality ---------- #


def initialize_ground_state_adapter(gpw, calc, context, *,
                                    dynamic_ground_state):
    if dynamic_ground_state:
        gs = GSAdapterWithPAWCache(calc)
    else:
        gs = GSAdapterWithPAWCache.from_gpw_file(gpw, context=context)

    return gs


class GSAdapterWithPAWCache(ResponseGroundStateAdapter):
    """Add a PAW correction cache to the ground state adapter.

    WARNING: Use with extreme care! The cache is only valid, when the
    plane-wave representation is kept fixed.
    """

    def __init__(self, calc):
        super().__init__(calc)

        self._pair_density_paw_corrections = []

    def pair_density_paw_corrections(self, qpd):
        """Overwrite method with a cached version."""
        for q_c, pwpaw_corr_data in self._pair_density_paw_corrections:
            if np.allclose(qpd.q_c, q_c):
                return pwpaw_corr_data

        pwpaw_corr_data = super().pair_density_paw_corrections(qpd)
        self._pair_density_paw_corrections.append((qpd.q_c, pwpaw_corr_data))

        return pwpaw_corr_data


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
    for chi1_GG, chi2_GG in zip(chiks_q[q1].array, chiks_q[q2].array):
        # Check the reciprocity
        assert chi2_GG[invmap_GG].T == pytest.approx(chi1_GG, rel=rtol,
                                                     abs=1e-8)
        # Check inversion symmetry
        assert chi2_GG[invmap_GG] == pytest.approx(chi1_GG, rel=rtol, abs=1e-8)

    # Loop over q-vectors
    for chiks in chiks_q:
        for chiks_GG in chiks.array:  # array = chiks_zGG
            # Check that the full susceptibility matrix is symmetric
            assert chiks_GG.T == pytest.approx(chiks_GG, rel=rtol, abs=1e-8)


def compare_dsnb_pw_bases(chiks_dsnbq, dsnb1, dsnb2):
    chiks1_q, chiks2_q = take_two_dsnb_settings(chiks_dsnbq, dsnb1, dsnb2)
    compare_pw_bases(chiks1_q, chiks2_q)


def compare_pw_bases(chiks1_q, chiks2_q):
    """Compare the plane-wave representations of two calculated chiks."""
    for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
        G1_Gc = get_pw_coordinates(chiks1.qpd)
        G2_Gc = get_pw_coordinates(chiks2.qpd)
        assert G1_Gc.shape == G2_Gc.shape
        assert np.allclose(G1_Gc - G2_Gc, 0.)


def compare_dsnb_arrays(chiks_dsnbq, dsnb1, dsnb2, *, rtol):
    chiks1_q, chiks2_q = take_two_dsnb_settings(chiks_dsnbq, dsnb1, dsnb2)
    compare_arrays(chiks1_q, chiks2_q, rtol=rtol)


def compare_arrays(chiks1_q, chiks2_q, *, rtol):
    """Compare the values inside two arrays."""
    for chiks1, chiks2 in zip(chiks1_q, chiks2_q):
        assert chiks2.array == pytest.approx(chiks1.array, rel=rtol, abs=1e-8)


def take_two_dsnb_settings(chiks_dsnbq, dsnb1, dsnb2):
    d1, s1, n1, b1 = dsnb1
    d2, s2, n2, b2 = dsnb2
    chiks1_q = chiks_dsnbq[d1][s1][n1][b1]
    chiks2_q = chiks_dsnbq[d2][s2][n2][b2]

    return chiks1_q, chiks2_q
