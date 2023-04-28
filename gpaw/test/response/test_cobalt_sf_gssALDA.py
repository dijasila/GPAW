# General modules
import pytest
import numpy as np

# Script modules
from gpaw import GPAW
from gpaw.test import findpeak
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import (ChiFactory, spectral_decomposition,
                                          EigendecomposedSpectrum)
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.dyson import HXCScaling
from gpaw.response.pair_functions import read_susceptibility_array


@pytest.mark.kspair
@pytest.mark.response
def test_response_cobalt_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    fxc = 'ALDA'
    q_qc = [[0.0, 0.0, 0.0], [1. / 4., 0.0, 0.0]]  # Two q-points along G-M
    frq_w = np.linspace(-0.2, 1.2, 57)
    eta = 0.2

    rshelmax = 0
    hxc_scaling = HXCScaling('fm')
    ecut = 250
    reduced_ecut = 100  # ecut for eigenmode analysis
    pos_eigs = 2  # majority modes
    neg_eigs = 0  # minority modes
    nblocks = 'max'

    # ---------- Script ---------- #

    # Initialize objects to calculat Chi
    context = ResponseContext()
    calc = GPAW(gpw_files['co_pw_wfs'], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']
    gs = ResponseGroundStateAdapter(calc)
    chiks_calc = ChiKSCalculator(gs, context,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
        gs, context, rshelmax=rshelmax)
    chi_factory = ChiFactory(chiks_calc, fxc_calculator)

    for q, q_c in enumerate(q_qc):
        complex_frequencies = frq_w + 1.j * eta
        chiks, chi = chi_factory('+-', q_c, complex_frequencies,
                                 fxc=fxc, hxc_scaling=hxc_scaling)
        chiks = chiks.copy_with_reduced_ecut(reduced_ecut)
        Aksmaj, _ = spectral_decomposition(chiks, pos_eigs=0, neg_eigs=0)
        Aksmaj.write(f'Aksmaj_q{q}.pckl')
        chi = chi.copy_with_reduced_ecut(reduced_ecut)
        chi.write_diagonal(f'chiwG_q{q}.pckl')
        Amaj, _ = spectral_decomposition(chi,
                                         pos_eigs=pos_eigs,
                                         neg_eigs=neg_eigs)
        Amaj.write(f'Amaj_q{q}.pckl')
        assert f'{fxc},+-' in chi_factory.fxc_kernel_cache

    context.write_timer()
    context.comm.barrier()

    # Read data
    w0_w, _, chi0_wG = read_susceptibility_array('chiwG_q0.pckl')
    w1_w, _, chi1_wG = read_susceptibility_array('chiwG_q1.pckl')
    Amaj0 = EigendecomposedSpectrum.from_file('Amaj_q0.pckl')
    Amaj1 = EigendecomposedSpectrum.from_file('Amaj_q1.pckl')
    Aksmaj0 = EigendecomposedSpectrum.from_file('Aksmaj_q0.pckl')
    Aksmaj1 = EigendecomposedSpectrum.from_file('Aksmaj_q1.pckl')

    # Find acoustic magnon mode
    wpeak00, Ipeak00 = findpeak(w0_w, -chi0_wG[:, 0].imag)
    wpeak01, Ipeak01 = findpeak(w1_w, -chi1_wG[:, 0].imag)
    # Find optical magnon mode
    wpeak10, Ipeak10 = findpeak(w0_w, -chi0_wG[:, 1].imag)
    wpeak11, Ipeak11 = findpeak(w1_w, -chi1_wG[:, 1].imag)

    # Use the eigenvectors at the acoustic magnon peaks to distinguish the
    # two magnon modes
    w0 = np.argmin(np.abs(Amaj0.omega_w - wpeak00))
    s00_w = extract_magnon_eigenmodes(Amaj0, w0, 0)
    s10_w = extract_magnon_eigenmodes(Amaj0, w0, 1)
    w1 = np.argmin(np.abs(Amaj1.omega_w - wpeak01))
    s01_w = extract_magnon_eigenmodes(Amaj1, w1, 0)
    s11_w = extract_magnon_eigenmodes(Amaj1, w1, 1)

    # Find peaks in eigenmodes
    mpeak00, Speak00 = findpeak(Amaj0.omega_w, s00_w)
    mpeak01, Speak01 = findpeak(Amaj1.omega_w, s01_w)
    mpeak10, Speak10 = findpeak(Amaj0.omega_w, s10_w)
    mpeak11, Speak11 = findpeak(Amaj1.omega_w, s11_w)

    # Calculate the full spectral enhancement at the acoustic magnon maxima
    enh0 = Amaj0.A_w[w0] / Aksmaj0.A_w[w0]
    enh1 = Amaj1.A_w[w1] / Aksmaj1.A_w[w1]

    if context.comm.rank == 0:
        # import matplotlib.pyplot as plt
        # # Plot the magnon lineshapes
        # # q=0
        # plt.subplot(2, 2, 1)
        # plt.plot(w0_w, -chi0_wG[:, 0].imag)
        # plt.axvline(wpeak00, c='0.5', linewidth=0.8)
        # plt.plot(w0_w, -chi0_wG[:, 1].imag)
        # plt.axvline(wpeak10, c='0.5', linewidth=0.8)
        # plt.plot(Amaj0.omega_w, Amaj0.s_we[:, 0])
        # plt.plot(Amaj0.omega_w, Amaj0.s_we[:, 1])
        # plt.plot(Amaj0.omega_w, s00_w)
        # plt.plot(Amaj0.omega_w, s10_w)
        # # q=1
        # plt.subplot(2, 2, 2)
        # plt.plot(w1_w, -chi1_wG[:, 0].imag)
        # plt.axvline(wpeak01, c='0.5', linewidth=0.8)
        # plt.plot(w1_w, -chi1_wG[:, 1].imag)
        # plt.axvline(wpeak11, c='0.5', linewidth=0.8)
        # plt.plot(Amaj1.omega_w, Amaj1.s_we[:, 0])
        # plt.plot(Amaj1.omega_w, Amaj1.s_we[:, 1])
        # plt.plot(Amaj1.omega_w, s01_w)
        # plt.plot(Amaj1.omega_w, s11_w)
        # # Plot full spectral weight
        # # q=0
        # plt.subplot(2, 2, 3)
        # plt.plot(Amaj0.omega_w, Amaj0.A_w)
        # plt.plot(Aksmaj0.omega_w, Aksmaj0.A_w)
        # plt.axvline(wpeak00, c='0.5', linewidth=0.8)
        # # q=1
        # plt.subplot(2, 2, 4)
        # plt.plot(Amaj1.omega_w, Amaj1.A_w)
        # plt.plot(Aksmaj1.omega_w, Aksmaj1.A_w)
        # plt.axvline(wpeak01, c='0.5', linewidth=0.8)
        # plt.show()

        # Print values
        print(hxc_scaling.lambd)
        print(wpeak00, wpeak01, wpeak10, wpeak11)
        print(Ipeak00, Ipeak01, Ipeak10, Ipeak11)
        print(mpeak00, mpeak01, mpeak10, mpeak11)
        print(Speak00, Speak01, Speak10, Speak11)
        print(enh0, enh1)

    # Test kernel scaling
    assert hxc_scaling.lambd == pytest.approx(0.9859, abs=0.005)

    # Test magnon frequencies
    assert wpeak00 == pytest.approx(-0.0279, abs=0.005)
    assert wpeak01 == pytest.approx(0.224, abs=0.01)
    assert wpeak10 == pytest.approx(0.851, abs=0.01)
    assert wpeak11 == pytest.approx(0.671, abs=0.01)

    # Test magnon amplitudes
    assert Ipeak00 == pytest.approx(2.895, abs=0.01)
    assert Ipeak01 == pytest.approx(2.209, abs=0.01)
    assert Ipeak10 == pytest.approx(1.013, abs=0.01)
    assert Ipeak11 == pytest.approx(0.934, abs=0.01)

    # Test magnon frequency consistency
    assert mpeak00 == pytest.approx(wpeak00, abs=0.005)
    assert mpeak01 == pytest.approx(wpeak01, abs=0.01)
    assert mpeak10 == pytest.approx(wpeak10, abs=0.01)
    assert mpeak11 == pytest.approx(wpeak11, abs=0.01)

    # Test magnon mode eigenvalues at extrema
    assert Speak00 == pytest.approx(6.085, abs=0.01)
    assert Speak01 == pytest.approx(4.796, abs=0.01)
    assert Speak10 == pytest.approx(2.587, abs=0.01)
    assert Speak11 == pytest.approx(2.426, abs=0.01)

    # Test enhancement factors
    assert enh0 == pytest.approx(46.18, abs=0.1)
    assert enh1 == pytest.approx(31.87, abs=0.1)


def extract_magnon_eigenmodes(Amaj, windex, eindex):
    v_G = Amaj.v_wGe[windex, :, eindex]
    s_w = np.conj(v_G) @ Amaj.A_wGG @ v_G
    assert np.allclose(s_w.imag, 0.)
    return s_w.real
