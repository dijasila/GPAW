import time
import pytest
import numpy as np

from ase import Atoms
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.test import findpeak
from gpaw.bztools import find_high_symmetry_monkhorst_pack
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.mpi import size, world


@pytest.mark.response
def test_response_Na_EELS_RPA_tetra_point_comparison(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation
    t1 = time.time()

    a = 3
    atom = Atoms('Na',
                 cell = [a, 0, 0],
                 pbc = [1, 1, 1])
    atom.center(vacuum = 1 * a, axis = (1, 2))
    atom.center()
    atoms = atom.repeat((2, 1, 1))

    calc = GPAW(mode=PW(200),
                nbands=4,
                setups = {'Na': '1'},
                kpts=(16, 2, 2),
                parallel={'band': 1},
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Na_gs')

    # Generate grid compatible with tetrahedron integration
    kpts = find_high_symmetry_monkhorst_pack('Na_gs', 6.0)

    # Calculate the wave functions on the new kpts grid
    calc = GPAW('Na_gs').fixed_density(kpts=kpts)
    # calc.get_potential_energy()
    calc.write('Na', 'all')

    t2 = time.time()

    # Excited state calculation
    q0_c = np.array([0., 0., 0.])
    q1_c = np.array([1 / 4., 0., 0.])
    w_w = np.linspace(0, 24, 241)

    # Calculate the eels spectrum using point integration at both q-points
    df1 = DielectricFunction(calc='Na', frequencies=w_w, eta=0.2, ecut=50,
                             hilbert=False, rate=0.2)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Na-PI_q0', q_c=q0_c)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Na-PI_q1', q_c=q1_c)

    t3 = time.time()

    # Calculate the eels spectrum using tetrahedron integration at q=0
    # NB: We skip the finite q-point, because the underlying symmetry
    # exploration runs excruciatingly slowly at finite q...
    df2 = DielectricFunction(calc='Na', eta=0.2, ecut=50,
                             integrationmode='tetrahedron integration',
                             hilbert=True, rate=0.2)
    df2.get_eels_spectrum(xc='RPA', filename='EELS_Na-TI_q0', q_c=q0_c)

    t4 = time.time()

    parprint('')
    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For PI excited state calc, it took', (t3 - t2) / 60, 'minutes')
    parprint('For TI excited state calc, it took', (t4 - t3) / 60, 'minutes')

    world.barrier()
    omegaP0_w, eels0P0_w, eelsP0_w = read_response_function('EELS_Na-PI_q0')
    omegaP1_w, eels0P1_w, eelsP1_w = read_response_function('EELS_Na-PI_q1')
    omegaT0_w, eels0T0_w, eelsT0_w = read_response_function('EELS_Na-TI_q0')

    # New results are compared with test values
    wpeak1P0, Ipeak1P0 = findpeak(omegaP0_w, eels0P0_w)
    wpeak2P0, Ipeak2P0 = findpeak(omegaP0_w, eelsP0_w)
    wpeak1P1, Ipeak1P1 = findpeak(omegaP1_w, eels0P1_w)
    wpeak2P1, Ipeak2P1 = findpeak(omegaP1_w, eelsP1_w)

    # New results are compared with test values
    wpeak1T0, Ipeak1T0 = findpeak(omegaT0_w, eels0T0_w)
    wpeak2T0, Ipeak2T0 = findpeak(omegaT0_w, eelsT0_w)

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.plot(omegaP0_w, eelsP0_w)
    # plt.plot(omegaT0_w, eelsT0_w)
    # plt.show()

    # tetra and point integrators should produce similar results:
    # confirm this by comparing the 2 integration methods
    assert wpeak1T0 == pytest.approx(wpeak1P0, abs = 0.05)
    assert wpeak2T0 == pytest.approx(wpeak2P0, abs = 0.05)

    # ensure the wpeak for point & tetra integration do not change
    assert wpeak1P0 == pytest.approx(3.4811, abs=0.02)
    assert wpeak2P0 == pytest.approx(3.4811, abs=0.02)
    assert wpeak1P1 == pytest.approx(3.7166, abs=0.02)
    assert wpeak2P1 == pytest.approx(3.8076, abs=0.02)
    assert wpeak1T0 == pytest.approx(3.5033, abs=0.02)
    assert wpeak2T0 == pytest.approx(3.5033, abs=0.02)

    # ensure the Ipeak for point & tetra integration do not change
    assert Ipeak1P0 == pytest.approx(8.6311, abs=1.)
    assert Ipeak2P0 == pytest.approx(8.6311, abs=1.)
    assert Ipeak1P1 == pytest.approx(8.4380, abs=1.)
    assert Ipeak2P1 == pytest.approx(7.7766, abs=1.)
    assert Ipeak1T0 == pytest.approx(8.3715, abs=1.)
    assert Ipeak2T0 == pytest.approx(8.3715, abs=1.)
