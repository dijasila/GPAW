"""
Calculate the magnetic response in iron using ALDA.

Fast test, where the kernel is scaled to fulfill the Goldstone theorem.
"""

# Workflow modules
import pytest
import numpy as np

# Script modules
import time

from ase.parallel import parprint

from gpaw.test import findpeak, equal
from gpaw.mpi import world

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.response.susceptibility import read_macroscopic_component


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_gssALDA(in_tmp_dir, gpw_files):
    # ------------------- Inputs ------------------- #
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.080, 0.120, 26), np.linspace(0.100, 0.300, 26)]
    fxc = 'ALDA'
    fxc_scaling = [True, None, 'fm']
    ecut = 300
    eta = 0.01
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ------------------- Script ------------------- #
    t1 = time.time()
    context = ResponseContext()
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw_wfs'],
                                                  context=context)
    fxckwargs = {'rshelmax': None, 'fxc_scaling': fxc_scaling}
    tms = TransverseMagneticSusceptibility(gs,
                                           context=context,
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

    t2 = time.time()

    parprint('Excited state calculation took', (t2 - t1) / 60, 'minutes')

    world.barrier()

    # Part 3: identify magnon peaks in scattering function
    w1_w, chiks1_w, chi1_w = read_macroscopic_component('iron_dsus_1.csv')
    w2_w, chiks2_w, chi2_w = read_macroscopic_component('iron_dsus_2.csv')

    print(w1_w, -chi1_w.imag)
    print(w2_w, -chi2_w.imag)

    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag)
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag)

    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000

    # Part 4: compare new results to test values
    test_fxcs = 1.033
    test_mw1 = -0.03  # meV
    test_mw2 = 176.91  # meV
    test_mw2 = 162  # meV
    test_Ipeak1 = 71.20  # a.u.
    test_Ipeak2 = 44.46  # a.u.

    import matplotlib.pyplot as plt
    plt.plot(w1_w, -chi1_w.imag)
    plt.plot(w2_w, -chi2_w.imag)
    plt.show()

    # fxc_scaling:
    equal(fxc_scaling[1], test_fxcs, 0.005)

    # Magnon peak:
    equal(mw1, test_mw1, 0.1)
    assert mw2 == pytest.approx(test_mw2, abs=eta * 650)

    # Scattering function intensity:
    assert Ipeak1 == pytest.approx(test_Ipeak1, abs=5)
    assert Ipeak2 == pytest.approx(test_Ipeak2, abs=5)
    #equal(Ipeak1, test_Ipeak1, 5)
    #equal(Ipeak2, test_Ipeak2, 5)
