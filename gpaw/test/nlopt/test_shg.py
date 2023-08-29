import pytest
import numpy as np
from gpaw import GPAW, PW
from ase import Atoms
from gpaw.nlopt.shg import get_shg
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.mpi import world
from ase.io import read


@pytest.mark.later
@pytest.mark.skipif(world.size > 4, reason='System too small')
def test_shg(in_tmp_dir):
    # Check for Hydrogen atom
    atoms = Atoms('H', cell=(3 * np.eye(3)), pbc=True)

    # Do a GS and save it
    calc = GPAW(
        mode=PW(600), symmetry={'point_group': False},
        kpts={'size': (2, 2, 2)}, nbands=5, txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', 'all')

    # Get the mml
    make_nlodata()

    # Do a SHG
    get_shg(freqs=np.linspace(0, 5, 100))
    get_shg(freqs=np.linspace(0, 5, 100), gauge='vg', out_name='shg2.npy')

    # Check it
    if world.rank == 0:
        shg = np.load('shg.npy')
        shg2 = np.load('shg2.npy')
        # Check for nan's
        assert not np.isnan(shg).any()
        assert not np.isnan(shg2).any()
        # Check the two gauges
        assert np.all(np.abs(shg2[1] - shg[1]) < 1e-3)
        # It should be zero (small) since H2 is centro-symm.
        assert np.all(np.abs(shg[1]) < 1e-8)


def test_shg_spinpol(in_tmp_dir):
    # Check Black Phosphorus for non-centro-symm test.
    atoms = Atoms('P4', positions=[[0.03948480, -0.00027057, 7.49990646],
                                   [0.86217564, -0.00026338, 9.60988536],
                                   [2.35547782, 1.65277230, 9.60988532],
                                   [3.17816857, 1.65277948, 7.49990643]],
                  cell=[4.63138807675, 3.306178252090, 17.10979291],
                  pbc=[True, True, False])
    atoms.center(vacuum=4, axis=2)

    # Do a spinpaired GS and save it
    calc = GPAW(mode=PW(250),
                symmetry={'point_group': False,
                          'time_reversal': True,
                          'symmorphic': False},
                spinpol=False,
                xc='LDA',
                kpts={'size': (4, 4, 1), 'gamma': True},
                nbands=15,
                convergence={'density': 1.0e-8, 'bands': -3}, txt='gs.txt')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', 'all')

    # Get the mml
    make_nlodata()

    # Do a SHG for yyy and xxz
    get_shg(freqs=np.linspace(1, 5, 50), pol='yyy',
            out_name='shg_yyy_nospinpol.npy')
    get_shg(freqs=np.linspace(1, 5, 50), pol='xxz',
            out_name='shg_xxz_nospinpol.npy')

    # Repeat steps for spinpol and save it
    calc = GPAW(mode=PW(250),
                symmetry={'point_group': False,
                          'time_reversal': True,
                          'symmorphic': False},
                spinpol=True,
                xc='LDA',
                kpts={'size': (4, 4, 1), 'gamma': True},
                nbands=15,
                convergence={'density': 1.0e-8, 'bands': -3}, txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', 'all')

    # Get the mml
    make_nlodata()

    # Do a SHG
    get_shg(freqs=np.linspace(1, 5, 50), pol='yyy',
            out_name='shg_yyy_spinpol.npy')
    get_shg(freqs=np.linspace(1, 5, 50), pol='xxz',
            out_name='shg_xxz_spinpol.npy')

    atoms = read('gs.txt')
    cellsize = atoms.cell.cellpar()
    
    # Make the sheet susceptibility from z-direction cellsize
    mult = cellsize[2] * 1e-10

    # Check spinpol vs nospinpol for yyy
    if world.rank == 0:
        shg = np.load('shg_yyy_nospinpol.npy')
        shg2 = np.load('shg_yyy_spinpol.npy')
        # Check for nan's
        assert not np.isnan(shg).any()
        assert not np.isnan(shg2).any()
        # Check the two SHG.
        real_shg = np.real(shg[1]) * mult * 1e18
        imag_shg = np.imag(shg[1]) * mult * 1e18
        real_shg2 = np.real(shg2[1]) * mult * 1e18
        imag_shg2 = np.imag(shg2[1]) * mult * 1e18
        
        # Assert difference between spinpol and nospinpol is very small.
        # The response spectra is of the order 1e-4,
        # so we check the difference till 1e-7.
        assert np.all(np.abs(real_shg - real_shg2) < 1e-7)
        assert np.all(np.abs(imag_shg - imag_shg2) < 1e-7)

        # Check spinpol vs nospinpol for xxz
        shg = np.load('shg_xxz_nospinpol.npy')
        shg2 = np.load('shg_xxz_spinpol.npy')
        # Check for nan's
        assert not np.isnan(shg).any()
        assert not np.isnan(shg2).any()
        # Check the two SHG.
        real_shg = np.real(shg[1]) * mult * 1e18
        imag_shg = np.imag(shg[1]) * mult * 1e18
        real_shg2 = np.real(shg2[1]) * mult * 1e18
        imag_shg2 = np.imag(shg2[1]) * mult * 1e18
        
        # Assert difference between spinpol and nospinpol is zero very small.
        assert np.all(np.abs(real_shg - real_shg2) < 1e-7)
        assert np.all(np.abs(imag_shg - imag_shg2) < 1e-7)
