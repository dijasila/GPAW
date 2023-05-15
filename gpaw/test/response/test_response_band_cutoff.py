import pytest
import numpy as np
from gpaw.test.conftest import response_band_cutoff
from gpaw import GPAW
import gpaw.mpi as mpi

@pytest.mark.response
def test_response_band_cutoff(in_tmp_dir, gpw_files):

    for gs, nbands in response_band_cutoff.items():
        calc = GPAW(gpw_files[gs],
                    communicator=mpi.serial_comm)
        nconv = calc.parameters.convergence['bands']
        #assert nbands < nconv


def get_nbands_cutoff_list(wfs, nconv):
    
