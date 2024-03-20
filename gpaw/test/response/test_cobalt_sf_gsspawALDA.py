import pytest

import numpy as np

from gpaw import GPAW
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator, SelfEnhancementCalculator
from gpaw.response.dyson import DysonEnhancer, ScaledDysonEnhancer
from gpaw.response.susceptibility import (spectral_decomposition,
                                          read_eigenmode_lineshapes)

from gpaw.test import findpeak
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.kspair
@pytest.mark.response
def test_response_cobalt_sf_gsspawALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    q_qc = [[0.0, 0.0, 0.0], [1. / 4., 0.0, 0.0]]  # Two q-points along G-M
    frq_w = np.linspace(-0.5, 2.0, 101)
    eta = 0.2

    rshelmax = 0
    ecut = 150
    pos_eigs = 5
    nmodes = 2  # majority modes
    nblocks = 'max'

    # ---------- Script ---------- #

    # Read ground state data
    context = ResponseContext(txt='cobalt_susceptibility.txt')
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    gs = ResponseGroundStateAdapter(calc)

    # Set up response calculators
    calc_args = (gs,)
    calc_kwargs = dict(context=context,
                       nbands=nbands,
                       ecut=ecut,
                       gammacentered=True,
                       bandsummation='pairwise',
                       nblocks=nblocks)
    chiks_calc = ChiKSCalculator(*calc_args, **calc_kwargs)
    xi_calc = SelfEnhancementCalculator(*calc_args,
                                        rshelmax=rshelmax,
                                        **calc_kwargs)
    dyson_enhancer = DysonEnhancer(context)
    lambd = None

    for q, q_c in enumerate(q_qc):
        # Calculate χ_KS^+-(q,z) and Ξ^++(q,z)
        zd = ComplexFrequencyDescriptor.from_array(frq_w + 1j * eta)
        chiks = chiks_calc.calculate('+-', q_c, zd)
        xi = xi_calc.calculate('+-', q_c, zd)

        # Distribute frequencies and invert dyson equation
        chiks = chiks.copy_with_global_frequency_distribution()
        xi = xi.copy_with_global_frequency_distribution()
        chi = dyson_enhancer(chiks, xi)
        # Test ability to apply Goldstone scaling when inverting the dyson eq.
        scaled_dyson_enhancer = ScaledDysonEnhancer.from_xi_calculator(
            xi_calc, lambd=lambd)
        scaled_chi = scaled_dyson_enhancer(chiks, xi)
        lambd = scaled_dyson_enhancer.lambd

        # Calculate majority spectral function
        Amaj, _ = spectral_decomposition(chi, pos_eigs=pos_eigs)
        Amaj.write_eigenmode_lineshapes(
            f'cobalt_Amaj_q{q}.csv', nmodes=nmodes)
        sAmaj, _ = spectral_decomposition(scaled_chi, pos_eigs=pos_eigs)
        sAmaj.write_eigenmode_lineshapes(
            f'cobalt_sAmaj_q{q}.csv', nmodes=nmodes)

        # Store Re ξ^++(q=0,ω), to test the effectiveness of the scaling
        if q == 0:
            chiks_mW, xi_mW = get_mode_projections(
                chiks, xi, sAmaj, lambd=lambd, nmodes=nmodes)
            xi0_w = xi_mW[0].real

        # plot_enhancement(chiks, xi, Amaj, sAmaj, lambd=lambd, nmodes=nmodes)

    context.write_timer()

    # Compare scaling coefficient to reference
    assert lambd == pytest.approx(1.0685, abs=0.001)
    # Test that 1 - Re ξ^++(q=0,ω) vanishes at ω=0
    w0 = np.argmin(np.abs(frq_w))
    assert xi0_w[w0] == pytest.approx(1., abs=0.001)

    # Compare magnon peaks to reference data
    refs_mqa = [
        # Acoustic
        [
            # q_Γ
            [
                # (wpeak, Apeak)
                (0.085, 7.895),  # unscaled
                (-0.026, 8.550),  # scaled
            ],
            # q_M / 2
            [
                # (wpeak, Apeak)
                (0.320, 5.828),  # unscaled
                (0.226, 6.752),  # scaled
            ],
        ],
        # Optical
        [
            # q_Γ
            [
                # (wpeak, Apeak)
                (0.904, 3.493),  # unscaled
                (0.524, 3.689),  # scaled
            ],
            # q_M / 2
            [
                # (wpeak, Apeak)
                (0.857, 2.988),  # unscaled
                (0.693, 3.455),  # scaled
            ],
        ],
    ]
    for a, Astr in enumerate(['Amaj', 'sAmaj']):
        for q in range(len(q_qc)):
            w_w, a_wm = read_eigenmode_lineshapes(f'cobalt_{Astr}_q{q}.csv')
            for m in range(nmodes):
                wpeak, Apeak = findpeak(w_w, a_wm[:, m])
                refw, refA = refs_mqa[m][q][a]
                print(m, q, a, wpeak, Apeak)
                assert wpeak == pytest.approx(refw, abs=0.01)  # eV
                assert Apeak == pytest.approx(refA, abs=0.05)  # a.u.


def get_mode_projections(chiks, xi, Amaj, *, lambd, nmodes):
    """Project χ_KS^+-(q,z) and Ξ^++(q,z) onto the magnon mode vectors."""
    wm = Amaj.get_eigenmode_frequency(nmodes=nmodes)
    v_Gm = Amaj.get_eigenvectors_at_frequency(wm, nmodes=nmodes)
    chiks_wm = np.zeros((chiks.blocks1d.nlocal, nmodes), dtype=complex)
    xi_wm = np.zeros((xi.blocks1d.nlocal, nmodes), dtype=complex)
    for m, v_G in enumerate(v_Gm.T):
        chiks_wm[:, m] = np.conj(v_G) @ chiks.array @ v_G  # chiks_wGG
        xi_wm[:, m] = np.conj(v_G) @ xi.array @ v_G  # xi_wGG
    chiks_mW = chiks.blocks1d.all_gather(chiks_wm * lambd).T
    xi_mW = xi.blocks1d.all_gather(xi_wm * lambd).T
    return chiks_mW, xi_mW


def plot_enhancement(chiks, xi, Amaj0, sAmaj, *, lambd, nmodes):
    import matplotlib.pyplot as plt
    from gpaw.mpi import world
    from ase.units import Ha

    for Amaj, _lambd in zip([Amaj0, sAmaj], [1., lambd]):
        a_mW = Amaj.get_eigenmode_lineshapes(nmodes=nmodes).T
        chiks_mW, xi_mW = get_mode_projections(
            chiks, xi, Amaj, lambd=_lambd, nmodes=nmodes)
        for m in range(nmodes):
            plt.subplot(1, nmodes, m + 1)
            plt.plot(chiks.zd.omega_w * Ha, -chiks_mW[m].imag / np.pi)
            plt.plot(xi.zd.omega_w * Ha, xi_mW[m].real)
            plt.axhline(1., c='0.5')
            plt.plot(Amaj.omega_w, a_mW[m])

    if world.rank == 0:
        plt.show()
    world.barrier()
