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
        ('fe_pw_wfs', '+-', 0.04, 0.02, 0.02)
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


def generate_nblocks_n():
    nblocks_n = [1]
    if world.size % 2 == 0:
        nblocks_n.append(2)
    if world.size % 4 == 0:
        nblocks_n.append(4)

    return nblocks_n


# ---------- Actual tests ---------- #


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
    """

    # ---------- Inputs ---------- #

    # Part 1: Set up ChiKSTestingFactory
    wfs, spincomponent, rtol, dsym_rtol, bsum_rtol = system
    q_c = get_q_c(wfs, qrel)

    ecut = 50
    # Test vanishing and finite real and imaginary frequencies
    frequencies = np.array([0., 0.05, 0.1, 0.2])
    complex_frequencies = list(frequencies + 0.j) + list(frequencies + 0.1j)
    zd = ComplexFrequencyDescriptor.from_array(complex_frequencies)

    # Part 2: Check toggling of calculation parameters
    # Note: None of these should change the actual results.
    disable_syms_s = [True, False]

    nblocks_n = generate_nblocks_n()
    nn = len(nblocks_n)

    bandsummation_b = ['double', 'pairwise']
    bundle_integrals_i = [True, False]

    nblocks_rtol = 1e-6
    bint_rtol = 1e-6

    # Part 3: Check reciprocity and inversion symmetry

    # ---------- Script ---------- #

    # Part 1: Set up ChiKSTestingFactory
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']

    chiks_testing_factory = ChiKSTestingFactory(calc,
                                                spincomponent, q_c, zd,
                                                nbands, ecut, gammacentered)

    # Part 2: Check toggling of calculation parameters

    # Check symmetry toggle and cross-validate with nblocks and bandsummation
    for nblocks in nblocks_n:
        for bandsummation in bandsummation_b:
            chiks1 = chiks_testing_factory(disable_syms=False,
                                           qsign=1,
                                           bundle_integrals=True,
                                           nblocks=nblocks,
                                           bandsummation=bandsummation)
            chiks2 = chiks_testing_factory(disable_syms=True,
                                           qsign=1,
                                           bundle_integrals=True,
                                           nblocks=nblocks,
                                           bandsummation=bandsummation)
            compare_pw_bases(chiks1, chiks2)
            compare_arrays(chiks1, chiks2, rtol=dsym_rtol)

    # Check nblocks and cross-validate with disable_syms and bandsummation
    for disable_syms in disable_syms_s:
        for bandsummation in bandsummation_b:
            for n1, n2 in combinations(range(nn), 2):
                chiks1 = chiks_testing_factory(nblocks=nblocks_n[n1],
                                               qsign=1,
                                               bundle_integrals=True,
                                               disable_syms=disable_syms,
                                               bandsummation=bandsummation)
                chiks2 = chiks_testing_factory(nblocks=nblocks_n[n2],
                                               qsign=1,
                                               bundle_integrals=True,
                                               disable_syms=disable_syms,
                                               bandsummation=bandsummation)
                compare_pw_bases(chiks1, chiks2)
                compare_arrays(chiks1, chiks2, rtol=nblocks_rtol)

    # Check bandsummation and cross-validate with disable_syms and nblocks
    for disable_syms in disable_syms_s:
        for nblocks in nblocks_n:
            chiks1 = chiks_testing_factory(bandsummation='double',
                                           qsign=1,
                                           bundle_integrals=True,
                                           disable_syms=disable_syms,
                                           nblocks=nblocks)
            chiks2 = chiks_testing_factory(bandsummation='pairwise',
                                           qsign=1,
                                           bundle_integrals=True,
                                           disable_syms=disable_syms,
                                           nblocks=nblocks)
            compare_pw_bases(chiks1, chiks2)
            compare_arrays(chiks1, chiks2, rtol=bsum_rtol)

    # Check bundle_integrals toggle and cross-validate with nblocks
    for nblocks in nblocks_n:
        chiks1 = chiks_testing_factory(bundle_integrals=True,
                                       qsign=1,
                                       disable_syms=False,
                                       nblocks=nblocks,
                                       bandsummation='pairwise')
        chiks1 = chiks_testing_factory(bundle_integrals=False,
                                       qsign=1,
                                       disable_syms=False,
                                       nblocks=nblocks,
                                       bandsummation='pairwise')
        compare_pw_bases(chiks1, chiks2)
        compare_arrays(chiks1, chiks2, rtol=bint_rtol)

    # Part 3: Check reciprocity and inversion symmetry

    # Cross-tabulate disable_syms, nblocks and bandsummation
    for disable_syms in disable_syms_s:
        for nblocks in nblocks_n:
            for bandsummation in bandsummation_b:
                # Calculate chiks for q and -q
                chiks1 = chiks_testing_factory(qsign=1,
                                               bundle_integrals=True,
                                               disable_syms=disable_syms,
                                               nblocks=nblocks,
                                               bandsummation=bandsummation)
                if np.allclose(q_c, 0.):
                    chiks2 = chiks1
                else:
                    chiks2 = chiks_testing_factory(qsign=-1,
                                                   bundle_integrals=True,
                                                   disable_syms=disable_syms,
                                                   nblocks=nblocks,
                                                   bandsummation=bandsummation)
                check_reciprocity_and_inversion_symmetry(chiks1, chiks2,
                                                         rtol=rtol)

    # Cross-tabulate bundle_integrals and nblocks
    for bundle_integrals in bundle_integrals_i:
        for nblocks in nblocks_n:
            # Calculate chiks for q and -q
            chiks1 = chiks_testing_factory(
                qsign=1,
                bundle_integrals=bundle_integrals, nblocks=nblocks,
                disable_syms=False, bandsummation='pairwise')
            if np.allclose(q_c, 0.):
                chiks2 = chiks1
            else:
                chiks2 = chiks_testing_factory(
                    qsign=-1,
                    bundle_integrals=bundle_integrals, nblocks=nblocks,
                    disable_syms=False, bandsummation='pairwise')
            check_reciprocity_and_inversion_symmetry(chiks1, chiks2, rtol=rtol)


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


class ChiKSTestingFactory:
    """Factory to calculate and cache ChiKS objects."""

    def __init__(self, calc,
                 spincomponent, q_c, zd,
                 nbands, ecut, gammacentered):
        self.gs = GSAdapterWithPAWCache(calc)
        self.spincomponent = spincomponent
        self.q_c = q_c
        self.zd = zd
        self.nbands = nbands
        self.ecut = ecut
        self.gammacentered = gammacentered

        self.cached_chiks = {}

    def __call__(self, *, qsign: int,
                 bundle_integrals: bool, disable_syms: bool,
                 bandsummation: str, nblocks: int):
        # Compile a string of the calculation parameters for cache look-up
        cache_string = f'{qsign},{bundle_integrals},{disable_syms}'
        cache_string += f'{bandsummation},{nblocks}'

        if cache_string in self.cached_chiks:
            return self.cached_chiks[cache_string]
        
        chiks_calc = ChiKSCalculator(
            self.gs, ecut=self.ecut, nbands=self.nbands,
            gammacentered=self.gammacentered,
            bundle_integrals=bundle_integrals,
            disable_time_reversal=disable_syms,
            disable_point_group=disable_syms,
            bandsummation=bandsummation,
            nblocks=nblocks)

        chiks = chiks_calc.calculate(
            self.spincomponent, qsign * self.q_c, self.zd)
        chiks = chiks.copy_with_global_frequency_distribution()
        self.cached_chiks[cache_string] = chiks

        return chiks


class GSAdapterWithPAWCache(ResponseGroundStateAdapter):
    """Add a PAW correction cache to the ground state adapter.

    WARNING: Use with care! The cache is only valid, when the plane-wave
    representations are identical.
    """

    def __init__(self, calc):
        super().__init__(calc)

        self._cached_corrections = []
        self._cached_parameters = []

    def pair_density_paw_corrections(self, qpd):
        """Overwrite method with a cached version."""
        cache_index = self._cache_lookup(qpd)
        if cache_index:
            return self._cached_corrections[cache_index]

        return self._calculate_correction(qpd)

    def _calculate_correction(self, qpd):
        correction = super().pair_density_paw_corrections(qpd)

        self._cached_corrections.append(correction)
        self._cached_parameters.append((qpd.q_c, qpd.ecut, qpd.gammacentered))

        return correction

    def _cache_lookup(self, qpd):
        for i, (q_c, ecut,
                gammacentered) in enumerate(self._cached_parameters):
            if np.allclose(qpd.q_c, q_c) and abs(qpd.ecut - ecut) < 1e-8\
               and qpd.gammacentered == gammacentered:
                # Cache hit!
                return i


def compare_pw_bases(chiks1, chiks2):
    """Compare the plane-wave representations of two calculated chiks."""
    G1_Gc = get_pw_coordinates(chiks1.qpd)
    G2_Gc = get_pw_coordinates(chiks2.qpd)
    assert G1_Gc.shape == G2_Gc.shape
    assert np.allclose(G1_Gc - G2_Gc, 0.)


def compare_arrays(chiks1, chiks2, *, rtol):
    """Compare the values inside two chiks arrays."""
    assert chiks2.array == pytest.approx(chiks1.array, rel=rtol, abs=1e-8)


def check_reciprocity_and_inversion_symmetry(chiks1, chiks2, *, rtol):
    """Check the calculated susceptibility for reciprocity and inversion symmetry

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
    the ground state calculation.
    """
    invmap_GG = get_inverted_pw_mapping(chiks1.qpd, chiks2.qpd)

    # Loop over frequencies
    for chi1_GG, chi2_GG in zip(chiks1.array, chiks2.array):
        # Check the reciprocity
        assert chi2_GG[invmap_GG].T == pytest.approx(chi1_GG, rel=rtol,
                                                     abs=1e-8)
        # Check inversion symmetry
        assert chi2_GG[invmap_GG] == pytest.approx(chi1_GG, rel=rtol, abs=1e-8)

    # Loop over q-vectors
    for chiks in [chiks1, chiks2]:
        for chiks_GG in chiks.array:  # array = chiks_zGG
            # Check that the full susceptibility matrix is symmetric
            assert chiks_GG.T == pytest.approx(chiks_GG, rel=rtol, abs=1e-8)
