import pytest
import numpy as np
from ase import Atoms

from gpaw import GPAW, PW
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


def test_shg_spinpol(gpw_files, in_tmp_dir):
    freqs = np.linspace(2, 4, 101)
    shg_xyz = {}
    for spinpol in ['spinpaired', 'spinpol']:
        tag = '_spinpol' if spinpol == 'spinpol' else ''

        # Get nlodata from pre-calculated SiC fixtures
        calc = gpw_files[f'sic_pw{tag}']
        make_nlodata(calc, out_name=f'mml{tag}.npz')
        world.barrier()

        # Calculate 'xyz' tensor element of SHG spectra
        get_shg(freqs=freqs, eta=0.025, pol='xyz',
                out_name=f'shg_xyz{tag}.npy',
                mml_name=f'mml{tag}.npz')
        world.barrier()

        # Load the calculated SHG spectra (in units of nm/V)
        shg_xyz[str(spinpol)] = np.load(f'shg_xyz{tag}.npy')[1] * 1e9

    # import matplotlib.pyplot as plt
    # plt.plot(freqs, shg_xyz['spinpaired'])
    # plt.plot(freqs, shg_xyz['spinpol'])
    # plt.show()

    # Assert that the difference between spectra from spinpaired and
    # spinpolarised calculations is small

    # Absolute error
    shg_xyz_diff = shg_xyz['spinpaired'] - shg_xyz['spinpol']
    assert shg_xyz_diff.real == pytest.approx(0, abs=1e-3)
    assert shg_xyz_diff.imag == pytest.approx(0, abs=1e-3)

    # Relative error
    shg_xyz_avg = (shg_xyz['spinpaired'] + shg_xyz['spinpol']) / 2
    shg_xyz_rerr_real = shg_xyz_diff.real / shg_xyz_avg.real
    shg_xyz_rerr_imag = shg_xyz_diff.imag / shg_xyz_avg.imag
    assert shg_xyz_rerr_real == pytest.approx(0, abs=2e-2), \
        np.max(np.abs(shg_xyz_rerr_real))
    assert shg_xyz_rerr_imag == pytest.approx(0, abs=2e-2), \
        np.max(np.abs(shg_xyz_rerr_imag))


def test_shg_hBN(gpw_files, in_tmp_dir):
    # SHG spectrum of h-BN as previously calculated.
    # We should be aware of any changes to this.
    shg_values = np.array([-498.32624277 - 69.02559975j,
                           -652.74329198 - 122.25657802j,
                           -939.87916222 - 269.17682137j,
                           -1539.46588908 - 919.4495916j,
                           391.40378777 - 3400.02524452j,
                           1408.14833639 - 749.79781681j,
                           859.34871229 - 237.75920108j,
                           597.45469429 - 112.27235265j,
                           453.42533737 - 64.8513409j,
                           363.29204261 - 42.20753326j])
    freqs = np.linspace(2, 2.4, 10)

    # Get nlodata from pre-calculated SiC fixtures
    calc = gpw_files['hbn_pw_nopg']
    make_nlodata(calc, out_name='mml.npz')
    world.barrier()

    # Calculate 'xyz' tensor element of SHG spectra
    get_shg(freqs=freqs, eta=0.025, pol='xyz',
            out_name='shg_xyz.npy',
            mml_name='mml.npz')
    world.barrier()

    # Load the calculated SHG spectra (in units of nm/V)
    shg_xyz = np.load('shg_xyz.npy')[1] * 1e15

    assert shg_xyz.real == pytest.approx(shg_values.real, abs=1e-4)
    assert shg_xyz.imag == pytest.approx(shg_values.imag, abs=1e-4)
