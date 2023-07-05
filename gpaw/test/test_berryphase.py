import numpy as np
from gpaw import GPAW
from gpaw.berryphase import get_polarization_phase, parallel_transport
from gpaw.berryphase import get_berry_phases
import gpaw.mpi as mpi
import pytest


def test_pol(in_tmp_dir, gpw_files):

    # It is ugly to convert to string. But this is required in
    # get_polarization_phase. Should be changed in the future...
    phi_c = get_polarization_phase(str(gpw_files['mos2_pw_nosym_wfs']))

    # Only should test modulo 2pi
    phi = phi_c / (2 * np.pi) % 1
    phitest = [6.60376287e-01, 3.39625036e-01, 0.0]
    assert np.allclose(phi, phitest, atol=1e-4)


def test_parallel_transport(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['mos2_pw_nosym_wfs'],
                communicator=mpi.serial_comm)
    # without SOC
    phi_km, S_km = parallel_transport(calc,
                                      direction=0,
                                      name='mos2', scale=0)
    phi_km = phi_km / (2 * np.pi) % 1
    phitest = [0.91475, 0.272581]
    phival = [phi_km[1, 12], phi_km[0, 0]]
    assert np.allclose(phival, phitest, atol=1e-3)
    # S_km corresponds to expectation value of spin-operator
    # so it is not well defined without soc, due to degeneracy
    # Therefore we only test the phases.
    
    # with SOC
    phi_km, S_km = parallel_transport(calc, direction=0,
                                      name='mos2', scale=1)
    phi_km = phi_km / (2 * np.pi) % 1

    # Test value of phase for some bands and k:s
    phitest = [0.91423, 0.27521]
    phival = [phi_km[1, 12], phi_km[0, 0]]
    assert np.allclose(phival, phitest, atol=1e-3)
    Stest = [0.99938, 0.99874]
    Sval = [S_km[1, 12], S_km[0, 0]]
    assert np.allclose(Sval, Stest, atol=1e-3)


def test_berry_phases(in_tmp_dir, gpw_files):

    calc = GPAW(gpw_files['mos2_pw_nosym_wfs'],
                communicator=mpi.serial_comm)

    ind, phases = get_berry_phases(calc)

    indtest = [[0, 6, 12, 18, 24, 30],
               [1, 7, 13, 19, 25, 31],
               [2, 8, 14, 20, 26, 32],
               [3, 9, 15, 21, 27, 33],
               [4, 10, 16, 22, 28, 34],
               [5, 11, 17, 23, 29, 35]]

    phasetest = [1.66179, 2.54985, 3.10069, 2.54985, 1.66179, 0.92385]
    assert np.allclose(ind, indtest)
    assert np.allclose(phases, phasetest, atol=1e-3)


def test_assertions(in_tmp_dir, gpw_files):
    """
    Functions should only work without symmetry
    Tests so that proper assertion is raised for calculator
    with symmetry enabled
    """

    gpw_file = gpw_files['mos2_pw_wfs']
    with pytest.raises(AssertionError):
        get_polarization_phase(str(gpw_file))

    calc = GPAW(gpw_file,
                communicator=mpi.serial_comm)

    with pytest.raises(AssertionError):
        ind, phases = get_berry_phases(calc)

    with pytest.raises(AssertionError):
        phi_km, S_km = parallel_transport(calc,
                                          direction=0,
                                          name='mos2', scale=0)
