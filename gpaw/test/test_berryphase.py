import numpy as np
from gpaw import GPAW
from gpaw.berryphase import get_polarization_phase
from gpaw.mpi import world
import gpaw.mpi as mpi
import pytest

def test_pol(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['mos2_pw'],
                communicator=mpi.serial_comm).fixed_density(
                    symmetry='off')
    calc.write('mos2+wfs.gpw', mode='all')

    phi_c = get_polarization_phase('mos2+wfs.gpw')
    # Only should test modulo 2pi
    phi = phi_c / (2 * np.pi) % 1
    print('phi',phi)
    phitest = [6.60376287e-01, 3.39625036e-01, 0.0]
    assert np.allclose(phi, phitest, atol=1e-4)
