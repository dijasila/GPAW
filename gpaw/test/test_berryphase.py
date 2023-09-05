import numpy as np
from gpaw import GPAW
from gpaw.berryphase import get_polarization_phase, parallel_transport
from gpaw.berryphase import get_berry_phases
import gpaw.mpi as mpi
import pytest


# Values from an earlier test
ref_phi_km = np.array(
    [[2.72907676e-04, 2.99369724e+00, 4.51932187e+00, 5.94725651e+00],
     [4.84334561e-03, 2.42519044e+00, 4.43335136e+00, 5.75115262e+00],
     [2.99682618e-02, 2.26119678e+00, 4.30480687e+00, 5.78042986e+00],
     [4.84334561e-03, 2.42519044e+00, 4.43335136e+00, 5.75115262e+00],
     [2.72907676e-04, 2.99369724e+00, 4.51932187e+00, 5.94725651e+00],
     [3.75847658e-03, 2.67197983e+00, 4.36511629e+00, 5.60446187e+00]])
    

def test_parallel_transport(in_tmp_dir, gpw_files):
    # Calculate the berry phases and spin projections
    gpw = gpw_files['mos2_pw_nosym']
    parallel_transport(str(gpw), name='mos2', scale=1)

    # Load phase-ordered data
    phi_km, S_km = load_renormalized_data('mos2')

    # Test against that the berry phases do not change (assuming that
    # they were correct to begin with)
    print(phi_km[:, ::7])
    assert phi_km[:, ::7] == pytest.approx(ref_phi_km, abs=0.05)


def load_renormalized_data(name):
    data = np.load(f'phases_{name}.npz')
    phi_km = data['phi_km']
    S_km = data['S_km']

    # Phases are only well-defined modulo 2pi
    phi_km %= 2 * np.pi

    # Sort bands by the berry phase
    indices = np.argsort(phi_km, axis=1)
    phi_km = np.take_along_axis(phi_km, indices, axis=1)
    S_km = np.take_along_axis(S_km, indices, axis=1)

    return phi_km, S_km


def test_pol(in_tmp_dir, gpw_files):

    # It is ugly to convert to string. But this is required in
    # get_polarization_phase. Should be changed in the future...
    phi_c = get_polarization_phase(str(gpw_files['mos2_pw_nosym']))

    # Only should test modulo 2pi
    phi = phi_c / (2 * np.pi)
    phitest = [6.60376287e-01, 3.39625036e-01, 0.0]
    err = phi - phitest
    assert err == pytest.approx(err.round(), abs=1e-3)


def test_berry_phases(in_tmp_dir, gpw_files):

    calc = GPAW(gpw_files['mos2_pw_nosym'],
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


# only master will raise, so this test will hang in parallel
@pytest.mark.serial
def test_assertions(in_tmp_dir, gpw_files):
    """
    Functions should only work without symmetry
    Tests so that proper assertion is raised for calculator
    with symmetry enabled
    """

    gpw_file = gpw_files['mos2_pw']
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
