import time
import pytest
import numpy as np

from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.test import findpeak
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.mpi import size, world


@pytest.mark.response
def test_response_aluminum_EELS_RPA(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation

    t1 = time.time()

    a = 4.043
    atoms = bulk('Al', 'fcc', a=a)
    atoms.center()
    calc = GPAW(mode=PW(200),
                nbands=4,
                kpts=(4, 4, 4),
                parallel={'band': 1},
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Al', 'all')
    t2 = time.time()

    # Excited state calculation
    q = np.array([1 / 4.0, 0, 0])
    w = np.linspace(0, 24, 241)

    df1 = DielectricFunction(calc='Al', frequencies=w, eta=0.2, ecut=50,
                             hilbert=False)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Al-PI', q_c=q)
     
    t3 = time.time()

    df2 = DielectricFunction(calc='Al', eta=0.2, ecut=50,
                             integrationmode='tetrahedron integration',
                             hilbert=True)
    df2.get_eels_spectrum(xc='RPA', filename='EELS_Al-TI', q_c=q)

    parprint('')
    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For excited state calc, it took', (t3 - t2) / 60, 'minutes')

    world.barrier()
    omegaP_w, eels0P_w, eelsP_w = read_response_function('EELS_Al-PI')
    omegaT_w, eels0T_w, eelsT_w = read_response_function('EELS_Al-TI')
    
    # New results are compared with test values
    wpeak1P, Ipeak1P = findpeak(omegaP_w, eels0P_w)
    wpeak2P, Ipeak2P = findpeak(omegaP_w, eelsP_w)
    
    # New results are compared with test values
    wpeak1T, Ipeak1T = findpeak(omegaT_w, eels0T_w)
    wpeak2T, Ipeak2T = findpeak(omegaT_w, eelsT_w)

    # XX tetra and point integrators should produce similar results; currently
    # they don't. For now test that TI results don't change, later compare
    # the wpeaks match
    assert pytest.approx([14.614087891386717, 14.61259708712905], 1e-2) == [
        wpeak1T, wpeak2T]
    assert pytest.approx([12.51675453510941, 11.858240221012624], 1) == [
        Ipeak1T, Ipeak2T]

    # plasmon peak check
    assert pytest.approx([15.7064968875, 15.728889329], 1e-2, abs=True) == [
        wpeak1P, wpeak2P]
    # check the spectrum strength
    assert pytest.approx([29.0721098689, 26.4625750021], 1, abs=True) == [
        Ipeak1P, Ipeak2P]
