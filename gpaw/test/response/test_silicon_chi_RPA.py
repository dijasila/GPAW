import numpy as np
import time

from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW, FermiDirac
from gpaw.test import findpeak, equal
from gpaw.response.df import DielectricFunction
from gpaw.response.susceptibility import FourComponentSusceptibilityTensor
from gpaw.response.susceptibility import read_macroscopic_component
from gpaw.mpi import size, world


def test_response_silicon_chi_RPA(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation

    t1 = time.time()

    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)
    atoms.center()
    calc = GPAW(mode=PW(200),
                nbands=8,
                kpts=(4, 4, 4),
                parallel={'domain': 1},
                idiotproof=False,  # allow uneven distribution of k-points
                occupations=FermiDirac(width=0.05),
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si', 'all')
    t2 = time.time()

    # Excited state calculation
    q = np.array([1 / 4.0, 0, 0])
    w = np.linspace(0, 24, 241)

    # Using DF
    df = DielectricFunction(calc='Si',
                            frequencies=w, eta=0.2, ecut=50,
                            hilbert=False)
    df.get_dynamic_susceptibility(xc='RPA', q_c=q, filename='Si_chi1.csv')

    t3 = time.time()

    world.barrier()

    # Using FCST
    fcst = FourComponentSusceptibilityTensor(calc, fxc='RPA',
                                             eta=0.2, ecut=50)
    fcst.get_macroscopic_component('00', q, w, filename='Si_chi2.csv')

    t4 = time.time()

    world.barrier()

    parprint('')
    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For excited state calc 1, it took', (t3 - t2) / 60, 'minutes')
    parprint('For excited state calc 2, it took', (t4 - t3) / 60, 'minutes')

    # The two response codes should hold identical results
    d1 = np.loadtxt('Si_chi1.csv', delimiter=',')
    wpeak1, Ipeak1 = findpeak(d1[:, 0], -d1[:, 4])
    w_w, chiks_w, chi_w = read_macroscopic_component('Si_chi2.csv')
    wpeak2, Ipeak2 = findpeak(w_w, chi_w.imag)

    equal(wpeak1, wpeak2, 0.02)
    equal(Ipeak1, Ipeak2, 1.0)
