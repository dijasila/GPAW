"""
Calculate the magnetic response in iron using ALDA.

Tests whether the magnon energies and scattering intensities
have changed for:
 * Different kernel calculation strategies
 * Different chi0 transitions summation strategies
"""

import time

import numpy as np
import pytest
from ase.parallel import parprint
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.susceptibility import read_macroscopic_component
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.test import findpeak

pytestmark = pytest.mark.skipif(world.size < 4, reason='world.size < 4')


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_ALDA(in_tmp_dir, scalapack, gpw_files):
    # ------------------- Inputs ------------------- #

    """
    # Part 1: ground state calculation
    xc = 'LDA'
    kpts = 4
    nb = 6
    pw = 300
    conv = {'density': 1.e-8,
            'forces': 1.e-8}
    a = 2.867
    mm = 2.21
    """

    # Part 2: magnetic response calculation
    q_c = [0.0, 0.0, 1 / 4.]
    fxc = 'ALDA'
    ecut = 300
    eta = 0.01

    # Test different kernel, summation and symmetry strategies
    # rshelmax, rshewmin, bandsummation, bundle_integrals, disable_syms
    strat_sd = [(None, None, 'pairwise', True, False),
                (-1, 0.001, 'pairwise', True, False),
                (-1, 0.001, 'pairwise', False, False),
                (-1, 0.000001, 'pairwise', True, False),
                (-1, 0.000001, 'double', True, False),
                (-1, 0.000001, 'double', True, True),
                (-1, None, 'pairwise', True, False),
                (3, None, 'pairwise', True, False)]
    frq_sw = [np.linspace(0.160, 0.320, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21),
              np.linspace(0.320, 0.480, 21)]

    # ------------------- Script ------------------- #

    # Part 1: ground state calculation

    t1 = time.time()

    calc = GPAW(gpw_files['fe_pw_wfs'], parallel=dict(domain=1))

    t2 = time.time()

    # Part 2: magnetic response calculation
    gs = ResponseGroundStateAdapter(calc)

    for s, ((rshelmax, rshewmin, bandsummation, bundle_integrals,
             disable_syms), frq_w) in enumerate(zip(strat_sd, frq_sw)):
        tms = TransverseMagneticSusceptibility(
            gs,
            fxc=fxc,
            eta=eta,
            ecut=ecut,
            bandsummation=bandsummation,
            fxckwargs={'rshelmax': rshelmax,
                       'rshewmin': rshewmin},
            bundle_integrals=bundle_integrals,
            disable_point_group=disable_syms,
            disable_time_reversal=disable_syms,
            nblocks=2)
        tms.get_macroscopic_component(
            '+-', q_c, frq_w,
            filename='iron_dsus' + '_G%d.csv' % (s + 1))
        tms.context.write_timer()

    t3 = time.time()

    parprint('Ground state calculation took', (t2 - t1) / 60, 'minutes')
    parprint('Excited state calculations took', (t3 - t2) / 60, 'minutes')

    world.barrier()

    # Part 3: identify magnon peak in scattering functions
    w1_w, chiks1_w, chi1_w = read_macroscopic_component('iron_dsus_G1.csv')
    w2_w, chiks2_w, chi2_w = read_macroscopic_component('iron_dsus_G2.csv')
    w3_w, chiks3_w, chi3_w = read_macroscopic_component('iron_dsus_G3.csv')
    w4_w, chiks4_w, chi4_w = read_macroscopic_component('iron_dsus_G4.csv')
    w5_w, chiks5_w, chi5_w = read_macroscopic_component('iron_dsus_G5.csv')
    w6_w, chiks6_w, chi6_w = read_macroscopic_component('iron_dsus_G6.csv')
    w7_w, chiks7_w, chi7_w = read_macroscopic_component('iron_dsus_G7.csv')
    w8_w, chiks8_w, chi8_w = read_macroscopic_component('iron_dsus_G8.csv')

    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag)
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag)
    wpeak3, Ipeak3 = findpeak(w3_w, -chi3_w.imag)
    wpeak4, Ipeak4 = findpeak(w4_w, -chi4_w.imag)
    wpeak5, Ipeak5 = findpeak(w5_w, -chi5_w.imag)
    wpeak6, Ipeak6 = findpeak(w6_w, -chi6_w.imag)
    wpeak7, Ipeak7 = findpeak(w7_w, -chi7_w.imag)
    wpeak8, Ipeak8 = findpeak(w8_w, -chi8_w.imag)

    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000
    mw3 = wpeak3 * 1000
    mw4 = wpeak4 * 1000
    mw5 = wpeak5 * 1000
    mw6 = wpeak6 * 1000
    mw7 = wpeak7 * 1000
    mw8 = wpeak8 * 1000

    # Part 4: compare new results to test values
    test_mw1 = 271.6  # meV
    test_mw2 = 399.9  # meV
    test_mw4 = 398.83  # meV
    test_Ipeak1 = 38.1  # a.u.
    test_Ipeak2 = 55.80  # a.u.
    test_Ipeak4 = 54.7  # a.u.

    # Different kernel strategies should remain the same
    # Magnon peak:
    assert mw1 == pytest.approx(test_mw1, abs=eta * 250)
    assert mw2 == pytest.approx(test_mw2, abs=eta * 250)
    assert mw4 == pytest.approx(test_mw4, abs=eta * 250)

    # Scattering function intensity:
    assert Ipeak1 == pytest.approx(test_Ipeak1, abs=2.5)
    assert Ipeak2 == pytest.approx(test_Ipeak2, abs=2.5)
    assert Ipeak4 == pytest.approx(test_Ipeak4, abs=2.5)

    # The bundled and unbundled integration methods should give the same
    assert mw2 == pytest.approx(mw3, abs=eta * 100)
    assert Ipeak2 == pytest.approx(Ipeak3, abs=1.0)

    # The two transitions summation strategies should give identical results
    assert mw4 == pytest.approx(mw5, abs=eta * 100)
    assert Ipeak4 == pytest.approx(Ipeak5, abs=1.0)

    # Toggling symmetry should preserve the result
    assert mw6 == pytest.approx(mw4, abs=eta * 100)
    assert Ipeak6 == pytest.approx(Ipeak4, abs=1.0)

    # Including vanishing coefficients should not matter for the result
    assert mw7 == pytest.approx(mw4, abs=eta * 100)
    assert Ipeak7 == pytest.approx(Ipeak4, abs=1.0)
    assert mw8 == pytest.approx(mw2, abs=eta * 100)
    assert Ipeak8 == pytest.approx(Ipeak2, abs=1.0)
