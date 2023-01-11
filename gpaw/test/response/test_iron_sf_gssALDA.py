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
from gpaw.response.localft import LocalGridFTCalculator, LocalPAWFTCalculator
from gpaw.response.fxc_kernels import FXCScaling
from gpaw.response.df import read_response_function


def set_up_fxc_calculators(gs, context):
    fxckwargs_and_identifiers = []

    # Set up old grid calculator (with file buffer)
    fxckwargs_old_grid = {'calculator': {'method': 'old',
                                         'rshelmax': None},
                          'filename': 'grid_ALDA_fxc.npy',
                          'fxc_scaling': FXCScaling('fm')}
    fxckwargs_and_identifiers.append((fxckwargs_old_grid, 'old_grid'))

    # Set up old paw calculator (without file buffer)
    fxckwargs_old_paw = {'calculator': {'method': 'old',
                                        'rshelmax': 0},
                         'fxc_scaling': FXCScaling('fm')}
    fxckwargs_and_identifiers.append((fxckwargs_old_paw, 'old_paw'))

    # Set up new grid calculator (without file buffer)
    localft_calc = LocalGridFTCalculator(gs, context)
    fxckwargs_new_grid = {'calculator': {'method': 'new',
                                         'localft_calc': localft_calc},
                          'fxc_scaling': FXCScaling('fm')}
    fxckwargs_and_identifiers.append((fxckwargs_new_grid, 'new_grid'))

    # Set up new paw calculator (with file buffer)
    localft_calc = LocalPAWFTCalculator(gs, context, rshelmax=0)
    fxckwargs_new_paw = {'calculator': {'method': 'new',
                                        'localft_calc': localft_calc},
                         'filename': 'paw_ALDA_fxc.npy',
                         'fxc_scaling': FXCScaling('fm')}
    fxckwargs_and_identifiers.append((fxckwargs_new_paw, 'new_paw'))

    return fxckwargs_and_identifiers


def get_test_values(identifier):
    test_mw1 = 0.  # meV
    test_Ipeak1 = 7.48  # a.u.
    if 'grid' in identifier:
        if 'old' in identifier:
            test_fxcs = 1.034
        else:  # new
            test_fxcs = 1.059
        test_mw2 = 363.  # meV
        test_Ipeak2 = 3.47  # a.u.
    else:  # paw
        if 'old' in identifier:
            test_fxcs = 1.129
        else:  # new
            test_fxcs = 1.131
        test_mw2 = 352.  # meV
        test_Ipeak2 = 3.35  # a.u.

    return test_fxcs, test_mw1, test_mw2, test_Ipeak1, test_Ipeak2


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    nbands = 6
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.080, 0.120, 26), np.linspace(0.250, 0.450, 26)]
    fxc = 'ALDA'
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

    fxckwargs_and_identifiers = set_up_fxc_calculators(gs, context)

    for q in range(2):
        complex_frequencies = frq_qw[q] + 1.j * eta
        # Calculate chi using the various fxc calculators
        for fxckwargs, identifier in fxckwargs_and_identifiers:
            chi = chi_factory('+-', q_qc[q], complex_frequencies,
                              fxc=fxc,
                              fxckwargs=fxckwargs)
            chi.write_macroscopic_component(identifier + '_iron_dsus'
                                            + '_%d.csv' % (q + 1))

            # Check that the fxc kernel exists as a file buffer, if applicable
            if 'filename' in fxckwargs:
                assert chi_factory.fxc_factory.file_buffer_exists(
                    fxckwargs['filename'])

        chi_factory.context.write_timer()

    world.barrier()

    # plot_comparison('old_grid', 'new_grid')
    # plot_comparison('old_paw', 'new_paw')

    # Compare results to test values
    for fxckwargs, identifier in fxckwargs_and_identifiers:
        fxcs = fxckwargs['fxc_scaling'].get_scaling()
        _, _, mw1, Ipeak1, _, _, mw2, Ipeak2 = extract_data(identifier)

        print(fxcs, mw1, mw2, Ipeak1, Ipeak2)

        (test_fxcs, test_mw1, test_mw2,
         test_Ipeak1, test_Ipeak2) = get_test_values(identifier)

        # fxc_scaling:
        assert fxcs == pytest.approx(test_fxcs, abs=0.005)

        # Magnon peak:
        assert mw1 == pytest.approx(test_mw1, abs=20.)
        assert mw2 == pytest.approx(test_mw2, abs=50.)

        # Scattering function intensity:
        assert Ipeak1 == pytest.approx(test_Ipeak1, abs=0.5)
        assert Ipeak2 == pytest.approx(test_Ipeak2, abs=0.5)


def extract_data(identifier):
    # Read data
    w1_w, chiks1_w, chi1_w = read_response_function(identifier
                                                    + '_iron_dsus_1.csv')
    w2_w, chiks2_w, chi2_w = read_response_function(identifier
                                                    + '_iron_dsus_2.csv')

    # Spectral function
    S1_w = -chi1_w.imag
    S2_w = -chi2_w.imag

    # Identify peaks
    wpeak1, Ipeak1 = findpeak(w1_w, S1_w)
    wpeak2, Ipeak2 = findpeak(w2_w, S2_w)

    # Peak positions in meV
    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000

    return w1_w, S1_w, mw1, Ipeak1, w2_w, S2_w, mw2, Ipeak2


def plot_comparison(identifier1, identifier2):
    w11_w, S11_w, _, _, w12_w, S12_w, _, _ = extract_data(identifier1)
    w21_w, S21_w, _, _, w22_w, S22_w, _, _ = extract_data(identifier2)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.plot(w11_w, S11_w)
    plt.plot(w21_w, S21_w)
    plt.subplot(1, 2, 2)
    plt.plot(w12_w, S12_w)
    plt.plot(w22_w, S22_w)
    plt.show()
