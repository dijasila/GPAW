import pytest
import numpy as np
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, FermiDirac, PW
from gpaw.mpi import world
from gpaw.test import findpeak
from gpaw.response.bse import BSE
from gpaw.response.df import read_response_function
from ase.units import Bohr


def create_bse(q_c=[0, 0, 0]):
    bse = BSE('MoS2.gpw',
              q_c=q_c,
              spinors=True,
              ecut=10,
              valence_bands=[8],
              conduction_bands=[9],
              eshift=0.8,
              nbands=15,
              write_h=False,
              write_v=False,
              wfile=None,
              mode='BSE',
              truncation='2D')
    return bse


@pytest.mark.response
def test_response_bse_MoS2_cut(in_tmp_dir, scalapack):
    calc = GPAW(mode=PW(180),
                xc='PBE',
                nbands='nao',
                setups={'Mo': '6'},
                occupations=FermiDirac(0.001),
                convergence={'bands': -5},
                kpts=(5, 5, 1))

    a = 3.1604
    c = 10.0

    cell = Hexagonal(symbol='Mo',
                     latticeconstant={'a': a, 'c': c}).get_cell()
    layer = Atoms(symbols='MoS2', cell=cell, pbc=(1, 1, 0),
                  scaled_positions=[(0, 0, 0),
                                    (2 / 3, 1 / 3, 0.3),
                                    (2 / 3, 1 / 3, -0.3)])

    pos = layer.get_positions()
    pos[1][2] = pos[0][2] + 3.172 / 2
    pos[2][2] = pos[0][2] - 3.172 / 2
    layer.set_positions(pos)
    layer.set_pbc([True, True, False])
    layer.center(axis=2)
    layer.calc = calc
    layer.get_potential_energy()
    calc.write('MoS2.gpw', mode='all')

    bse = create_bse()

    outw_w, outalpha_w = bse.get_polarizability(write_eig=None,
                                                eta=0.02,
                                                w_w=np.linspace(0., 5., 5001))
    world.barrier()
    w_w, alphareal_w, alphaimag_w = read_response_function('pol_bse.csv')

    # Check consistency with written results
    assert np.allclose(outw_w, w_w, atol=1e-5, rtol=1e-4)
    assert np.allclose(outalpha_w.real, alphareal_w, atol=1e-5, rtol=1e-4)
    assert np.allclose(outalpha_w.imag, alphaimag_w, atol=1e-5, rtol=1e-4)

    w0, I0 = findpeak(w_w[:1100], alphaimag_w[:1100])
    w1, I1 = findpeak(w_w[1100:1300], alphaimag_w[1100:1300])
    w1 += 1.1

    assert w0 == pytest.approx(0.58, abs=0.01)
    assert I0 == pytest.approx(38.8, abs=0.35)
    assert w1 == pytest.approx(2.22, abs=0.01)
    assert I1 == pytest.approx(6.3, abs=0.35)

    #################################################################
    # Absorption and EELS spectra for 2D materials should be identical
    # for q=0.
    #################################################################

    bse = create_bse()
    outw_w, eels = bse.get_eels_spectrum(w_w=w_w)

    bse = create_bse()
    pbc_c = bse.gs.pbc
    V = bse.gs.nonpbc_cell_product()
    factor = V * Bohr**(sum(~pbc_c)) / (4 * np.pi)
    outw_w, pol = bse.get_polarizability(w_w=w_w)
    assert np.allclose(pol.imag / factor, eels)

    #####################################################################
    # Absorption and EELS spectra for 2D materials should NOT be identical
    # for finite q.
    #####################################################################
    bse = create_bse(q_c=[0.2, 0.2, 0.0])
    outw_w, eels = bse.get_eels_spectrum(w_w=w_w)
    bse = create_bse(q_c=[0.2, 0.2, 0.0])
    outw_w, pol = bse.get_polarizability(w_w)

    assert not np.allclose(pol.imag / factor, eels)
