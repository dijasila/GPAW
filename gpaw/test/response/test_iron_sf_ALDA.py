"""
Calculate the magnetic response in iron using ALDA.

Tests whether the magnon energies and scattering intensities
have changed for:
 * Different kernel calculation strategies
 * Different chi0 transitions summation strategies
"""

import numpy as np
import pytest

import time

from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.response.susceptibility import read_macroscopic_component
from gpaw.test import findpeak, equal
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl


pytestmark = pytest.mark.skipif(
    world.size < 4 or not compiled_with_sl(),
    reason='world.size < 4 or not compiled_with_sl()')


def test_response_iron_sf_ALDA(in_tmp_dir):
    # ------------------- Inputs ------------------- #

    # Part 1: ground state calculation
    xc = 'LDA'
    kpts = 4
    nb = 6
    pw = 300
    conv = {'density': 1.e-8,
            'forces': 1.e-8}
    a = 2.867
    mm = 2.21

    # Part 2: magnetic response calculation
    q_c = [0.0, 0.0, 1 / 4.]
    fxc = 'ALDA'
    ecut = 300
    eta = 0.01

    # Test different kernel and summation strategies
    # rshelmax, rshewmin, bandsummation, bundle_integrals, bundle_kptpairs
    strat_sd = [(None, None, 'pairwise', True, True),
                (-1, 0.001, 'pairwise', True, True),
                (-1, 0.001, 'pairwise', False, True),
                (-1, 0.001, 'pairwise', True, False),
                (-1, 0.000001, 'pairwise', True, True),
                (-1, 0.000001, 'double', True, True),
                (-1, None, 'pairwise', True, True),
                (3, None, 'pairwise', True, True)]
    frq_sw = [np.linspace(0.160, 0.320, 21),
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

    Febcc = bulk('Fe', 'bcc', a=a)
    Febcc.set_initial_magnetic_moments([mm])

    calc = GPAW(xc=xc,
                mode=PW(pw),
                kpts=monkhorst_pack((kpts, kpts, kpts)),
                nbands=nb,
                convergence=conv,
                symmetry={'point_group': False},
                idiotproof=False,
                parallel={'domain': 1})

    Febcc.calc = calc
    Febcc.get_potential_energy()
    t2 = time.time()

    # Part 2: magnetic response calculation

    for s, ((rshelmax, rshewmin, bandsummation,
             bundle_integrals, bundle_kptpairs),
            frq_w) in enumerate(zip(strat_sd, frq_sw)):
        tms = TransverseMagneticSusceptibility(
            calc,
            fxc=fxc,
            eta=eta,
            ecut=ecut,
            bandsummation=bandsummation,
            fxckwargs={'rshelmax': rshelmax,
                       'rshewmin': rshewmin},
            bundle_integrals=bundle_integrals,
            bundle_kptpairs=bundle_kptpairs,
            nblocks=2)
        tms.get_macroscopic_component(
            '+-', q_c, frq_w,
            filename='iron_dsus' + '_G%d.csv' % (s + 1))
        tms.write_timer()

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

    wpeak1, Ipeak1 = findpeak(w1_w, chi1_w.imag)
    wpeak2, Ipeak2 = findpeak(w2_w, chi2_w.imag)
    wpeak3, Ipeak3 = findpeak(w3_w, chi3_w.imag)
    wpeak4, Ipeak4 = findpeak(w4_w, chi4_w.imag)
    wpeak5, Ipeak5 = findpeak(w5_w, chi5_w.imag)
    wpeak6, Ipeak6 = findpeak(w6_w, chi6_w.imag)
    wpeak7, Ipeak7 = findpeak(w7_w, chi7_w.imag)
    wpeak8, Ipeak8 = findpeak(w8_w, chi8_w.imag)

    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000
    mw3 = wpeak3 * 1000
    mw4 = wpeak4 * 1000
    mw5 = wpeak5 * 1000
    mw6 = wpeak6 * 1000
    mw7 = wpeak7 * 1000
    mw8 = wpeak8 * 1000

    # Part 4: compare new results to test values
    test_mw1 = 234.63  # meV
    test_mw2 = 397.33  # meV
    test_mw5 = 398.83  # meV
    test_Ipeak1 = 56.74  # a.u.
    test_Ipeak2 = 55.80  # a.u.
    test_Ipeak5 = 58.23  # a.u.

    # Different kernel strategies should remain the same
    # Magnon peak:
    equal(mw1, test_mw1, eta * 250)
    equal(mw2, test_mw2, eta * 250)
    equal(mw5, test_mw5, eta * 250)

    # Scattering function intensity:
    equal(Ipeak1, test_Ipeak1, 2.5)
    equal(Ipeak2, test_Ipeak2, 2.5)
    equal(Ipeak5, test_Ipeak5, 2.5)

    # The bundled and unbundled integration methods should give the same
    equal(mw2, mw3, eta * 100)
    equal(Ipeak2, Ipeak3, 1.0)

    # The bundled and unbundled data extraction should give the same result
    equal(mw2, mw4, eta * 100)
    equal(Ipeak2, Ipeak4, 1.0)

    # The two transitions summation strategies should give identical results
    equal(mw5, mw6, eta * 100)
    equal(Ipeak5, Ipeak6, 1.0)

    # Including vanishing coefficients should not matter for the result
    equal(mw7, mw5, eta * 100)
    equal(Ipeak7, Ipeak5, 1.0)
    equal(mw8, mw2, eta * 100)
    equal(Ipeak8, Ipeak2, 1.0)
