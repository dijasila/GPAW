import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE
from gpaw.test import findpeak, equal

pytestmark = pytest.mark.skipif(
    world.size != 4 or not compiled_with_sl(),
    reason='world.size != 4 or not compiled_with_sl()')


@pytest.mark.response
def test_response_bse_silicon(in_tmp_dir):

    w_ = 2.552
    I_ = 421.15
    eshift = 0.8
    a = 5.431  # From PRB 73,045112 (2006)

    # Standard calculation
    atoms = bulk('Si', 'diamond', a=a)
    atoms.positions -= a / 8
    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                nbands=12,
                convergence={'bands': -4})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si.gpw', 'all')

    bse = BSE('Si.gpw',
              ecut=50.,
              valence_bands=range(4),
              conduction_bands=range(4, 8),
              eshift=eshift,
              nbands=8,
              write_h=False,
              write_v=False)
    w_w, eps_w = bse.get_dielectric_function(filename=None,
                                             eta=0.2,
                                             w_w=np.linspace(0, 10, 2001))
    w, I = findpeak(w_w, eps_w.imag)
    equal(w, w_, 0.01)
    equal(I, I_, 0.1)

    # Test that soc code gives the same
    bse = BSE('Si.gpw',
              spinors=True,
              scale=0,
              ecut=50.,
              valence_bands=range(8),
              conduction_bands=range(8, 16),
              eshift=eshift,
              nbands=8,
              write_h=False,
              write_v=False)
    w_w, eps_w = bse.get_dielectric_function(filename=None,
                                             eta=0.2,
                                             w_w=np.linspace(0, 10, 2001))
    w, I = findpeak(w_w, eps_w.imag)
    equal(w, w_, 0.01)
    equal(I, I_, 0.1)

    # Calculation without symmetry
    atoms = bulk('Si', 'diamond', a=a)
    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                nbands=12,
                symmetry='off',
                convergence={'bands': -4})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si.gpw', 'all')
    
    bse = BSE('Si.gpw',
              ecut=50.,
              valence_bands=range(4),
              conduction_bands=range(4, 8),
              eshift=eshift,
              nbands=8,
              write_h=False,
              write_v=False)
    w_w, eps_w = bse.get_dielectric_function(filename=None,
                                             eta=0.2,
                                             w_w=np.linspace(0, 10, 2001))
    w, I = findpeak(w_w, eps_w.imag)
    equal(w, w_, 0.01)
    equal(I, I_, 0.1)
