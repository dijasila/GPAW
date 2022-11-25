import time
import pytest
import numpy as np

from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW, FermiDirac
from gpaw.test import findpeak, equal
from gpaw.mpi import size, world

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.df import DielectricFunction, read_response_function
# from gpaw.response.susceptibility import FourComponentSusceptibilityTensor
from gpaw.response.chiks import ChiKS
from gpaw.response.susceptibility import ChiFactory


@pytest.mark.kspair
@pytest.mark.response
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

    # Using the ChiFactory
    gs = ResponseGroundStateAdapter(calc)
    chiks = ChiKS(gs, eta=0.2, ecut=50)
    chi_factory = ChiFactory(chiks)
    chi = chi_factory('00', q, w, fxc='RPA')
    chi.write_macroscopic_component('Si_chi2.csv')

    t4 = time.time()

    world.barrier()

    parprint('')
    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For excited state calc 1, it took', (t3 - t2) / 60, 'minutes')
    parprint('For excited state calc 2, it took', (t4 - t3) / 60, 'minutes')

    # The two response codes should hold identical results
    w1_w, chiks1_w, chi1_w = read_response_function('Si_chi1.csv')
    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag)
    w2_w, chiks2_w, chi2_w = read_response_function('Si_chi2.csv')
    wpeak2, Ipeak2 = findpeak(w2_w, -chi2_w.imag)

    equal(wpeak1, wpeak2, 0.02)
    equal(Ipeak1, Ipeak2, 1.0)
