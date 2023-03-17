import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import world

@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('wfs',['fancy_si_pw_wfs',
                                'al_pw_wfs',
                                'fe_pw_wfs'])
def test_ibz2bz(in_tmp_dir, gpw_files, wfs):
    calc = GPAW(gpw_files[wfs])
