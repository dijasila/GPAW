import pytest
from gpaw.mpi import world
from gpaw.utilities import compiled_with_sl
import numpy as np
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, FermiDirac
from gpaw.test import findpeak, equal
from gpaw.response.bse import BSE

pytestmark = pytest.mark.skipif(
    world.size < 4 or not compiled_with_sl(),
    reason='world.size < 4 or not compiled_with_sl()')


def test_response_bse_magnon(in_tmp_dir):
    if 1:
        calc = GPAW(mode='pw',
                    xc='PBE',
                    nbands='nao',
                    occupations=FermiDirac(0.001),
                    convergence={'bands': -5},
                    kpts={'size': (6, 6, 1), 'gamma': True})

        a = 3.945
        c = 8.0

        cell = Hexagonal(symbol='Sc',
                         latticeconstant={'a': a, 'c': c}).get_cell()
        layer = Atoms(symbols='ScSe2', cell=cell, pbc=(1, 1, 1),
                      scaled_positions=[(0, 0, 0),
                                        (2 / 3, 1 / 3, 0.3),
                                        (2 / 3, 1 / 3, -0.3)])

        pos = layer.get_positions()
        pos[1][2] = pos[0][2] + 1.466
        pos[2][2] = pos[0][2] - 1.466
        layer.set_positions(pos)
        layer.set_initial_magnetic_moments([1.0, 0, 0])
        layer.calc = calc
        layer.get_potential_energy()
        calc.write('ScSe2.gpw', mode='all')

    bse = BSE('ScSe2.gpw',
              spinors=True,
              ecut=10,
              scale=0,
              valence_bands=[22],
              conduction_bands=[23],
              eshift=3.4,
              nbands=15,
              write_h=False,
              write_v=False,
              wfile=None,
              mode='BSE',
              truncation='2D')

    w_w = np.linspace(-2, 2, 4001)
    chi_Gw = bse.get_magnetic_susceptibility(pbc=[True, True, False],
                                            eta=0.1,
                                            write_eig='chi+-.dat',
                                            w_w=w_w)
    
    w, I = findpeak(w_w, -chi_Gw[0].imag)
    equal(w, -0.0199, 0.001)
    equal(I, 5.62, 0.01)


    bse = BSE('ScSe2.gpw',
              spinors=True,
              ecut=10,
              scale=0,
              q_c=[1/3, 1/3, 0],
              valence_bands=[22],
              conduction_bands=[23],
              eshift=3.4,
              nbands=15,
              write_h=False,
              write_v=False,
              wfile=None,
              mode='BSE',
              truncation='2D')

    w_w = np.linspace(-2, 2, 4001)
    chi_Gw = bse.get_magnetic_susceptibility(pbc=[True, True, False],
                                            eta=0.1,
                                            write_eig='chi+-.dat',
                                            w_w=w_w)
    
    w, I = findpeak(w_w, -chi_Gw[0].imag)
    equal(w, -0.0155, 0.001)
    equal(I, 2.82, 0.01)
