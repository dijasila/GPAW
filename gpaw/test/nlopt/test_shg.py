import pytest
import numpy as np
from gpaw import GPAW, PW
from ase import Atoms
from gpaw.nlopt.shg import get_shg
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.mpi import world
from ase.lattice.hexagonal import Graphene

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
    #Check for hBN for non-centro-symm test.
    atoms = Graphene(symbol='B',
                     latticeconstant={'a' : 2.5, 'c' : 1.0 },
                     size=(1,1,1))
    atoms[0].symbol = 'N'
    atoms.pbc = (1,1,0)
    atoms.center(axis=2, vacuum=3.0)

    #Do a GS for nospinpol and save it
    calc = GPAW(mode=PW(400),
                symmetry={'point_group': False, 'time_reversal': True,
                          'symmorphic': False},
                spinpol=False,
                xc = 'LDA',
                kpts={'size': (6,6,1), 'gamma': True},
                nbands = 8, 
                convergence = {'bands':-3}, txt = None)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', 'all')

    # Get the mml
    make_nlodata()

    # Do a SHG
    get_shg(freqs=np.linspace(0, 5, 100), out_name='shg_nospinpol.npy')

    #Repeat steps for spinpol and save it
    calc = GPAW(mode=PW(400),
                symmetry={'point_group': False, 'time_reversal': True,
                          'symmorphic': False},
                spinpol=True,
                xc = 'LDA',
                kpts={'size': (6,6,1), 'gamma': True},
                nbands = 8, 
                convergence = {'bands':-3}, txt = None)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', 'all')

    # Get the mml
    make_nlodata()

    # Do a SHG
    get_shg(freqs=np.linspace(0, 5, 100), out_name='shg_spinpol.npy')

    mult = 6 * 1e-10  # Make the sheet susceptibility from vac size (z-direction cellsize). 

    # Check spinpol vs nospinpol
    if world.rank == 0:
        shg = np.load('shg_nospinpol.npy')
        shg2 = np.load('shg_spinpol.npy')
        # Check for nan's
        assert not np.isnan(shg).any()
        assert not np.isnan(shg2).any()
        # Check the two SHG.
        real_shg = np.real(mult * shg[1] * 1e18)
        imag_shg = np.imag(mult * shg[1] * 1e18)
        real_shg2 = np.real(mult * shg2[1] * 1e18)
        imag_shg2 = np.imag(mult * shg2[1] * 1e18)
        
        # Assert difference between spinpol and nospinpol is zero (very small)
        assert np.all(np.abs(real_shg[1] - real_shg2[1]) < 1e-8)
        assert np.all(np.abs(imag_shg[1] - imag_shg2[1]) < 1e-8)


