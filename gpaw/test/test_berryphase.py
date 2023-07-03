import numpy as np
from gpaw import GPAW
from gpaw.berryphase import get_polarization_phase, parallel_transport
import gpaw.mpi as mpi


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
    # Stest = [0.99011, 1.0000]
    # Sval = [S_km[1, 12], S_km[0, 0]]
    # assert np.allclose(Sval, Stest, atol=1e-3)
    
    # with SOC
    phi_km, S_km = parallel_transport(calc, direction=0,
                                      name='mos2', scale=1)
    phi_km = phi_km / (2 * np.pi) % 1

    # Test value of phase for some bands and k:s
    phitest = [0.91423, 0.27521]
    phival = [phi_km[1, 12], phi_km[0, 0]]
    assert np.allclose(phival, phitest, atol=1e-3)
    # Stest = [0.99938, 0.99874]
    # Sval = [S_km[1, 12], S_km[0, 0]]
    # assert np.allclose(Sval, Stest, atol=1e-3)
