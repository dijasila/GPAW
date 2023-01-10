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
    old_w1_w, old_chiks1_w, old_chi1_w = read_response_function('old_iron_dsus_1.csv')
    old_w2_w, old_chiks2_w, old_chi2_w = read_response_function('old_iron_dsus_2.csv')
    new_w1_w, new_chiks1_w, new_chi1_w = read_response_function('new_iron_dsus_1.csv')
    new_w2_w, new_chiks2_w, new_chi2_w = read_response_function('new_iron_dsus_2.csv')

    print(old_w1_w, -old_chi1_w.imag)
    print(old_w2_w, -old_chi2_w.imag)
    print(new_w1_w, -new_chi1_w.imag)
    print(new_w2_w, -new_chi2_w.imag)

    old_wpeak1, old_Ipeak1 = findpeak(old_w1_w, -old_chi1_w.imag)
    old_wpeak2, old_Ipeak2 = findpeak(old_w2_w, -old_chi2_w.imag)
    new_wpeak1, new_Ipeak1 = findpeak(new_w1_w, -new_chi1_w.imag)
    new_wpeak2, new_Ipeak2 = findpeak(new_w2_w, -new_chi2_w.imag)

    old_mw1 = old_wpeak1 * 1000
    old_mw2 = old_wpeak2 * 1000
    new_mw1 = new_wpeak1 * 1000
    new_mw2 = new_wpeak2 * 1000

    old_fxcs = fxc_scaling_old.get_scaling()
    new_fxcs = fxc_scaling_new.get_scaling()

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.plot(old_w1_w, -old_chi1_w.imag)
    # plt.plot(new_w1_w, -new_chi1_w.imag)
    # plt.subplot(1, 2, 2)
    # plt.plot(old_w2_w, -old_chi2_w.imag)
    # plt.plot(new_w2_w, -new_chi2_w.imag)
    # plt.show()

    # Compare results to test values
    test_old_fxcs = 1.034
    test_new_fxcs = 1.059
    test_mw1 = 0.  # meV
    test_mw2 = 363.  # meV
    test_Ipeak1 = 7.48  # a.u.
    test_Ipeak2 = 3.47  # a.u.

    print(old_fxcs, old_mw1, old_mw2, old_Ipeak1, old_Ipeak2)
    print(new_fxcs, new_mw1, new_mw2, new_Ipeak1, new_Ipeak2)

    # fxc_scaling:
    assert old_fxcs == pytest.approx(test_old_fxcs, abs=0.005)
    assert new_fxcs == pytest.approx(test_new_fxcs, abs=0.005)

    # Magnon peak:
    assert old_mw1 == pytest.approx(test_mw1, abs=20.)
    assert old_mw2 == pytest.approx(test_mw2, abs=50.)
    assert new_mw1 == pytest.approx(test_mw1, abs=20.)
    assert new_mw2 == pytest.approx(test_mw2, abs=50.)

    # Scattering function intensity:
    assert old_Ipeak1 == pytest.approx(test_Ipeak1, abs=0.5)
    assert old_Ipeak2 == pytest.approx(test_Ipeak2, abs=0.5)
    assert new_Ipeak1 == pytest.approx(test_Ipeak1, abs=0.5)
    assert new_Ipeak2 == pytest.approx(test_Ipeak2, abs=0.5)
