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
    fxc_scaling = FXCScaling('fm')
    fxc_filename = 'ALDA_fxc.npy'
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
    fxckwargs = {'calculator': {'method': 'old',
                                'rshelmax': None},
                 'filename': fxc_filename,
                 'fxc_scaling': fxc_scaling,}
    chiks_calc = ChiKSCalculator(gs,
                                 context=context,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    chi_factory = ChiFactory(chiks_calc)

    for q in range(2):
        complex_frequencies = frq_qw[q] + 1.j * eta
        chi = chi_factory('+-', q_qc[q], complex_frequencies,
                          fxc=fxc,
                          fxckwargs=fxckwargs)

        # Check that the fxc kernel exists as a file buffer
        assert chi_factory.fxc_factory.file_buffer_exists(fxc_filename)

        chi.write_macroscopic_component('iron_dsus' + '_%d.csv' % (q + 1))
        chi_factory.context.write_timer()

    world.barrier()

    # Identify magnon peaks in scattering function
    w1_w, chiks1_w, chi1_w = read_response_function('iron_dsus_1.csv')
    w2_w, chiks2_w, chi2_w = read_response_function('iron_dsus_2.csv')

    print(w1_w, -chi1_w.imag)
    print(w2_w, -chi2_w.imag)

    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag)
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag)

    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000

    # Compare new results to test values
    test_fxcs = 1.034
    test_mw1 = 0.  # meV
    test_mw2 = 363.  # meV
    test_Ipeak1 = 7.48  # a.u.
    test_Ipeak2 = 3.47  # a.u.

    # fxc_scaling:
    fxcs = fxc_scaling.get_scaling()
    print(fxcs, mw1, mw2, Ipeak1, Ipeak2)
    assert fxcs == pytest.approx(test_fxcs, abs=0.005)

    # Magnon peak:
    assert mw1 == pytest.approx(test_mw1, abs=20.)
    assert mw2 == pytest.approx(test_mw2, abs=50.)

    # Scattering function intensity:
    assert Ipeak1 == pytest.approx(test_Ipeak1, abs=0.5)
    assert Ipeak2 == pytest.approx(test_Ipeak2, abs=0.5)
