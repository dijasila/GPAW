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
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.response.df import read_response_function


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    nbands = 6
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.080, 0.120, 26), np.linspace(0.250, 0.450, 26)]
    fxc = 'ALDA'
    fxc_scaling = [True, None, 'fm']
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
    fxckwargs = {'rshelmax': None, 'fxc_scaling': fxc_scaling}
    tms = TransverseMagneticSusceptibility(gs,
                                           context=context,
                                           nbands=nbands,
                                           fxc=fxc,
                                           eta=eta,
                                           ecut=ecut,
                                           fxckwargs=fxckwargs,
                                           gammacentered=True,
                                           nblocks=nblocks)

    for q in range(2):
        tms.get_macroscopic_component(
            '+-', q_qc[q], frq_qw[q],
            filename='iron_dsus' + '_%d.csv' % (q + 1))
        tms.context.write_timer()

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
    print(fxc_scaling[1], mw1, mw2, Ipeak1, Ipeak2)
    assert fxc_scaling[1] == pytest.approx(test_fxcs, abs=0.005)

    # Magnon peak:
    assert mw1 == pytest.approx(test_mw1, abs=20.)
    assert mw2 == pytest.approx(test_mw2, abs=50.)

    # Scattering function intensity:
    assert Ipeak1 == pytest.approx(test_Ipeak1, abs=1.)
    assert Ipeak2 == pytest.approx(test_Ipeak2, abs=1.)
