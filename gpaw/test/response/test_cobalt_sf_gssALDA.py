# General modules
import pytest
import numpy as np

# Script modules
from gpaw import GPAW
from gpaw.test import findpeak
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import ChiFactory, read_susceptibility_array
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.dyson import HXCScaling


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
    reduced_ecut = 10  # Only save the bare minimum
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
        _, chi = chi_factory('+-', q_c, complex_frequencies,
                             fxc=fxc, hxc_scaling=hxc_scaling)
        chi.write_reduced_diagonal(f'chiwG_q{q}.pckl',
                                   reduced_ecut=reduced_ecut)
        assert f'{fxc},+-' in chi_factory.fxc_kernel_cache

    context.write_timer()
    context.comm.barrier()

    # Read data
    w0_w, _, chi0_wG = read_susceptibility_array('chiwG_q0.pckl')
    w1_w, _, chi1_wG = read_susceptibility_array('chiwG_q1.pckl')

    # # Find acoustic magnon mode
    wpeak00, Ipeak00 = findpeak(w0_w, -chi0_wG[:, 0].imag)
    wpeak01, Ipeak01 = findpeak(w1_w, -chi1_wG[:, 0].imag)
    # Find optical magnon mode
    wpeak10, Ipeak10 = findpeak(w0_w, -chi0_wG[:, 1].imag)
    wpeak11, Ipeak11 = findpeak(w1_w, -chi1_wG[:, 1].imag)

    if context.comm.rank == 0:
        # # Plot the magnons
        # import matplotlib.pyplot as plt
        # # Acoustic magnon mode
        # plt.subplot(1, 2, 1)
        # plt.plot(w0_w, -chi0_wG[:, 0].imag)
        # plt.axvline(wpeak00, c='0.5', linewidth=0.8)
        # plt.plot(w1_w, -chi1_wG[:, 0].imag)
        # plt.axvline(wpeak01, c='0.5', linewidth=0.8)
        # # Optical magnon mode
        # plt.subplot(1, 2, 2)
        # plt.plot(w0_w, -chi0_wG[:, 1].imag)
        # plt.axvline(wpeak10, c='0.5', linewidth=0.8)
        # plt.plot(w1_w, -chi1_wG[:, 1].imag)
        # plt.axvline(wpeak11, c='0.5', linewidth=0.8)
        # plt.show()

        # Print values
        print(hxc_scaling.lambd)
        print(wpeak00, wpeak01, wpeak10, wpeak11)
        print(Ipeak00, Ipeak01, Ipeak10, Ipeak11)

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