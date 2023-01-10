"""
Calculate the magnetic response in iron using ALDA.

Fast test, where the kernel is scaled to fulfill the Goldstone theorem.
"""

# Workflow modules
import pytest
import numpy as np

# Script modules
from gpaw.test import findpeak
from gpaw.mpi import world

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import ChiFactory
from gpaw.response.localft import LocalGridFTCalculator
from gpaw.response.fxc_kernels import FXCScaling
from gpaw.response.df import read_response_function


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    nbands = 6
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.080, 0.120, 26), np.linspace(0.250, 0.450, 26)]
    fxc = 'ALDA'
    fxc_filename = 'ALDA_fxc.npy'
    fxc_scaling_old = FXCScaling('fm')
    fxc_scaling_new = FXCScaling('fm')
    ecut = 300
    eta = 0.1
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    context = ResponseContext()
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw_wfs'],
                                                  context=context)
    chiks_calc = ChiKSCalculator(gs,
                                 context=context,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    chi_factory = ChiFactory(chiks_calc)

    # Set up old and new fxc calculators
    fxckwargs_old = {'calculator': {'method': 'old',
                                    'rshelmax': None},
                     'filename': fxc_filename,
                     'fxc_scaling': fxc_scaling_old}
    localft_calc = LocalGridFTCalculator(gs, context)
    fxckwargs_new = {'calculator': {'method': 'new',
                                    'localft_calc': localft_calc},
                     'fxc_scaling': fxc_scaling_new}

    for q in range(2):
        complex_frequencies = frq_qw[q] + 1.j * eta
        # Calculate chi using the old fxc calculator
        chi = chi_factory('+-', q_qc[q], complex_frequencies,
                          fxc=fxc,
                          fxckwargs=fxckwargs_old)

        # Check that the fxc kernel exists as a file buffer
        assert chi_factory.fxc_factory.file_buffer_exists(fxc_filename)

        chi.write_macroscopic_component('old_iron_dsus' + '_%d.csv' % (q + 1))

        # Calculate chi using the new fxc calculator
        chi = chi_factory('+-', q_qc[q], complex_frequencies,
                          fxc=fxc,
                          fxckwargs=fxckwargs_new)

        chi.write_macroscopic_component('new_iron_dsus' + '_%d.csv' % (q + 1))

        chi_factory.context.write_timer()

    world.barrier()

    # Identify magnon peaks in scattering function
    w1_w, chiks1_w, chi1_w = read_response_function('old_iron_dsus_1.csv')
    w2_w, chiks2_w, chi2_w = read_response_function('old_iron_dsus_2.csv')
    w3_w, chiks3_w, chi3_w = read_response_function('new_iron_dsus_1.csv')
    w4_w, chiks4_w, chi4_w = read_response_function('new_iron_dsus_2.csv')

    print(w1_w, -chi1_w.imag)
    print(w2_w, -chi2_w.imag)
    print(w3_w, -chi3_w.imag)
    print(w4_w, -chi4_w.imag)

    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag)
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag)
    wpeak3, Ipeak3 = findpeak(w3_w, -chi3_w.imag)
    wpeak4, Ipeak4 = findpeak(w4_w, -chi4_w.imag)

    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000
    mw3 = wpeak3 * 1000
    mw4 = wpeak4 * 1000

    fxcs_old = fxc_scaling_old.get_scaling()
    fxcs_new = fxc_scaling_new.get_scaling()

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.plot(w1_w, -chi1_w.imag)
    # plt.plot(w3_w, -chi3_w.imag)
    # plt.subplot(1, 2, 2)
    # plt.plot(w2_w, -chi2_w.imag)
    # plt.plot(w4_w, -chi4_w.imag)
    # plt.show()

    # Compare results to test values
    test_fxcs_old = 1.034
    test_mw1 = 0.  # meV
    test_mw2 = 363.  # meV
    test_Ipeak1 = 7.48  # a.u.
    test_Ipeak2 = 3.47  # a.u.
    test_fxcs_new = 0.347
    test_mw3 = -4.5  # meV
    test_mw4 = 364.  # meV
    test_Ipeak3 = 7.47  # a.u.
    test_Ipeak4 = 3.35  # a.u.

    print(fxcs_old, mw1, mw2, Ipeak1, Ipeak2)
    print(fxcs_new, mw3, mw4, Ipeak3, Ipeak4)

    # fxc_scaling:
    assert fxcs_old == pytest.approx(test_fxcs_old, abs=0.005)
    assert fxcs_new == pytest.approx(test_fxcs_new, abs=0.005)

    # Magnon peak:
    assert mw1 == pytest.approx(test_mw1, abs=20.)
    assert mw2 == pytest.approx(test_mw2, abs=50.)
    assert mw3 == pytest.approx(test_mw3, abs=20.)
    assert mw4 == pytest.approx(test_mw4, abs=50.)

    # Scattering function intensity:
    assert Ipeak1 == pytest.approx(test_Ipeak1, abs=0.5)
    assert Ipeak2 == pytest.approx(test_Ipeak2, abs=0.5)
    assert Ipeak3 == pytest.approx(test_Ipeak3, abs=0.5)
    assert Ipeak4 == pytest.approx(test_Ipeak4, abs=0.5)
