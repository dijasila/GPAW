import pytest
import numpy as np
from gpaw import GPAW, PW
from ase import Atoms
from gpaw.nlopt.shg import get_shg
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.mpi import world


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
