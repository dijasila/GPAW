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

    def calculate_shg(spinpol):
        # Define common calculation parameters
        calc_params = {
            'mode': PW(250),
            'symmetry': {'point_group': False, 'time_reversal': True,
                         'symmorphic': False},
            'spinpol': spinpol,
            'xc': 'LDA',
            'kpts': {'size': (4, 4, 1), 'gamma': True},
            'nbands': 15,
            'convergence': {'density': 1.0e-8, 'bands': -3},
        }

        # Do a calculation and save it
        calc = GPAW(txt=None if spinpol else 'gs.txt', **calc_params)
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('gs.gpw', 'all')

        # Get the mml
        make_nlodata()

        # Do SHG for 'yyy' and 'xxz'
        shg_yyy = calculate_shg_polarization('yyy', spinpol)
        shg_xxz = calculate_shg_polarization('xxz', spinpol)

        return shg_yyy, shg_xxz

    def calculate_shg_polarization(pol, spinpol):
        out_name = f'shg_{pol}_{"spinpol" if spinpol else "nospinpol"}.npy'
        get_shg(freqs=np.linspace(1, 5, 100), pol=pol, out_name=out_name)
        return np.load(out_name)

    # Calculate SHG for 'yyy' and 'xxz' directions
    # for both spinpol and nospinpol
    shg_yyy_spinpol, shg_xxz_spinpol = calculate_shg(spinpol=True)
    shg_yyy_nospinpol, shg_xxz_nospinpol = calculate_shg(spinpol=False)
    
    atoms = read('gs.txt')
    cellsize = atoms.cell.cellpar()
    
    # Make the sheet susceptibility from z-direction cellsize
    mult = cellsize[2] * 1e-10
    # Calculate the scaling factor
    arrays_to_scale = [shg_xxz_spinpol[1], shg_xxz_nospinpol[1],
                       shg_yyy_spinpol[1], shg_yyy_nospinpol[1]]
    # Scale all arrays in the list
    for array in arrays_to_scale:
        array *= mult * 1e18

    assert not np.isnan(shg_yyy_spinpol).any()
    assert not np.isnan(shg_xxz_spinpol).any()
    assert not np.isnan(shg_yyy_nospinpol).any()
    assert not np.isnan(shg_xxz_nospinpol).any()

    # Check the difference between spinpol and nospinpol is very small
    assert np.all(np.abs(np.real(shg_yyy_spinpol[1])
                         - np.real(shg_yyy_nospinpol[1])) < 1e-8)
    assert np.all(np.abs(np.imag(shg_yyy_spinpol[1])
                         - np.imag(shg_yyy_nospinpol[1])) < 1e-8)
    assert np.all(np.abs(np.real(shg_xxz_spinpol[1])
                         - np.real(shg_xxz_nospinpol[1])) < 1e-8)
    assert np.all(np.abs(np.imag(shg_xxz_spinpol[1])
                         - np.imag(shg_xxz_nospinpol[1])) < 1e-8)
