import time
import pytest
import numpy as np

from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.test import findpeak, equal
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
    omega_wP, eels0_wP, eels_wP = read_response_function('EELS_Al-PI')
    omega_wT, eels0_wT, eels_wT = read_response_function('EELS_Al-TI')
    
    # New results are compared with test values
    wpeak1P, Ipeak1P = findpeak(omega_wP, eels0_wP)
    wpeak2P, Ipeak2P = findpeak(omega_wP, eels_wP)
    
    # New results are compared with test values
    wpeak1T, Ipeak1T = findpeak(omega_wT, eels0_wT)
    wpeak2T, Ipeak2T = findpeak(omega_wT, eels_wT)

    # XX tetra and point integrators should produce similar results; currently
    # they don't. For now test that TI results don't change, later compare
    # the wpeaks match
    test_wpeak1T = 14.614087891386717
    test_wpeak2T = 14.61259708712905
    test_Ipeak1T = 12.51675453510941
    test_Ipeak2T = 11.858240221012624
    equal(wpeak1T, test_wpeak1T, 1e-2)
    equal(wpeak2T, test_wpeak2T, 1e-2)
    equal(Ipeak1T, test_Ipeak1T, 1)
    equal(Ipeak2T, test_Ipeak2T, 1)

    test_wpeak1P = 15.7064968875  # eV
    test_Ipeak1P = 29.0721098689  # eV
    test_wpeak2P = 15.728889329  # eV
    test_Ipeak2P = 26.4625750021  # eV
    
    if np.abs(test_wpeak1P - wpeak1P) < 1e-2 and np.abs(test_wpeak2P -
                                                        wpeak2P) < 1e-2:
        pass
    else:
        print(test_wpeak1P - wpeak1P, test_wpeak2P - wpeak2P)
        raise ValueError('Plasmon peak not correct ! ')

    if abs(test_Ipeak1P - Ipeak1P) > 1 or abs(test_Ipeak2P - Ipeak2P) > 1:
        print((Ipeak1P - test_Ipeak1P, Ipeak2P - test_Ipeak2P))
        raise ValueError('Please check spectrum strength ! ')
