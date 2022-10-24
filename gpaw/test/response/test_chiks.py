"""Test functionality to compute the four-component susceptibility tensor for
the Kohn-Sham system."""

# General modules
import pytest
import numpy as np
from itertools import product

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world, rank

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.chiks import ChiKS
from gpaw.response.susceptibility import (get_pw_coordinates,
                                          get_inverted_pw_mapping)


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
def test_chiks_symmetry(in_tmp_dir, Fe_gs, q_c, eta, gammacentered):
    """Check the reciprocity relation,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G'-G)^(+-)(-q, ω),

    the inversion symmetry relation,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,-G-G')^(+-)(-q, ω),

    and the combination of the two,

    χ_(KS,GG')^(+-)(q, ω) = χ_(KS,G'G)^(+-)(q, ω),

    for a real life system with d-electrons and inversion symmetry (bcc-Fe).

    Unfortunately, there will always be random noise in the wave functions,
    such that these symmetries are not fulfilled exactly. However, we should be
    able to fulfill it within 2%, which is tested here. Generally speaking,
    the "symmetry" noise can be reduced making running with symmetry='off' in
    the ground state calculation.

    Also, we test that the response function does not change by toggling
    the symmetries within the same precision."""

    # ---------- Inputs ---------- #

    # Part 1: ChiKS calculation
    ecut = 50
    frequencies = [0., 0.05, 0.1, 0.2]
    disable_syms_s = [True, False]

    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # Part 2: Check reciprocity and inversion symmetry
    rtol = 2.5e-2

    # Part 3: Check matrix symmetry

    # Part 4: Check symmetry toggle
    srtol = 1.e-2

    # ---------- Script ---------- #

    # Part 1: ChiKS calculation
    calc, nbands = Fe_gs
    gs = ResponseGroundStateAdapter(calc)

    # Calculate chiks for q and -q
    if np.allclose(q_c, 0.):
        q_qc = [q_c]
    else:
        q_qc = [-q_c, q_c]

    chiks_sqwGG = []
    pd_sq = []
    for disable_syms in disable_syms_s:
        chiks = ChiKS(gs,
                      ecut=ecut, nbands=nbands, eta=eta,
                      gammacentered=gammacentered,
                      disable_time_reversal=disable_syms,
                      disable_point_group=disable_syms,
                      nblocks=nblocks)

        pd_q = []
        chiks_qwGG = []
        for q_c in q_qc:
            pd, chiks_wGG = chiks.calculate(q_c, frequencies,
                                            spincomponent='+-')
            chiks_wGG = chiks.distribute_frequencies(chiks_wGG)
            pd_q.append(pd)
            chiks_qwGG.append(chiks_wGG)

        chiks_sqwGG.append(chiks_qwGG)
        pd_sq.append(pd_q)

    # Part 2: Check reciprocity and inversion symmetry
    for pd_q, chiks_qwGG in zip(pd_sq, chiks_sqwGG):
        # Get the q and -q pair
        if len(pd_q) == 2:
            q1, q2 = 0, 1
        else:
            assert len(pd_q) == 1
            assert np.allclose(q_c, 0.)
            q1, q2 = 0, 0

        invmap_GG = get_inverted_pw_mapping(pd_q[q1], pd_q[q2])

        # Check reciprocity of the reactive part of the static
        # susceptibility. This specific check makes sure that the
        # exchange constants calculated within the MFT remains
        # reciprocal.
        if rank == 0:  # Only the root has the static susc.
            # Calculate the reactive part
            chi1_GG, chi2_GG = chiks_qwGG[q1][0], chiks_qwGG[q2][0]
            chi1r_GG = 1 / 2. * (chi1_GG + np.conj(chi1_GG).T)
            chi2r_GG = 1 / 2. * (chi2_GG + np.conj(chi2_GG).T)
            assert np.allclose(np.conj(chi2r_GG[invmap_GG]), chi1r_GG,
                               rtol=rtol)

        # Check the reciprocity of the full susceptibility
        for chi1_GG, chi2_GG in zip(chiks_qwGG[q1], chiks_qwGG[q2]):
            assert np.allclose(chi2_GG[invmap_GG].T, chi1_GG, rtol=rtol)

        # Check inversion symmetry of the full susceptibility
        for chi1_GG, chi2_GG in zip(chiks_qwGG[q1], chiks_qwGG[q2]):
            assert np.allclose(chi2_GG[invmap_GG], chi1_GG, rtol=rtol)

    # Part 3: Check matrix symmetry
    for chiks_qwGG in chiks_sqwGG:
        for chiks_wGG in chiks_qwGG:
            for chiks_GG in chiks_wGG:
                assert np.allclose(chiks_GG.T, chiks_GG, rtol=rtol)

    # Part 4: Check symmetry toggle

    # Check that the plane wave representations are identical
    for pd1, pd2 in zip(pd_sq[0], pd_sq[1]):
        G1_Gc = get_pw_coordinates(pd1)
        G2_Gc = get_pw_coordinates(pd2)
        assert G1_Gc.shape == G2_Gc.shape
        assert np.allclose(G1_Gc - G2_Gc, 0.)

    chiks1_qwGG = chiks_sqwGG[0]
    chiks2_qwGG = chiks_sqwGG[1]
    for chiks1_wGG, chiks2_wGG in zip(chiks1_qwGG, chiks2_qwGG):
        assert np.allclose(chiks2_wGG, chiks1_wGG, rtol=srtol)


# ---------- System ground state ---------- #


@pytest.fixture(scope='module')
def Fe_gs(module_tmp_path):

    # ---------- Inputs ---------- #

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

    # ---------- Script ---------- #

    atoms = bulk('Fe', 'bcc', a=a)
    atoms.set_initial_magnetic_moments([mm])
    atoms.center()

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                nbands=nbands + 4,
                occupations=FermiDirac(occw),
                parallel={'domain': 1},
                # symmetry='off',
                spinpol=True,
                convergence=conv
                )

    atoms.calc = calc
    atoms.get_potential_energy()

    return calc, nbands
