import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.nlopt.basic import NLOData
from gpaw.nlopt.shg import get_shg


def test_shg_spinpol(mme_files):
    shg_values = np.array([-0.77053399 - 0.37041593j,
                           -0.87903174 - 0.4177294j,
                           -1.00791251 - 0.51051291j,
                           -1.15465962 - 0.66642326j,
                           -1.30812094 - 0.92114822j,
                           -1.42138133 - 1.33424513j,
                           -1.34649601 - 1.96084827j,
                           -0.78891819 - 2.66240386j,
                           0.27801137 - 2.8572836j,
                           1.12315952 - 2.30446868j,
                           1.38569995 - 1.59698796j])

    freqs = np.linspace(2.3, 2.4, 11)
    shg_xyz = {}
    for spinpol in ['spinpaired', 'spinpol']:
        tag = '_spinpol' if spinpol == 'spinpol' else ''

        # Get pre-calculated nlodata from SiC fixtures
        nlodata = NLOData.load(mme_files[f'sic_pw{tag}'], comm=world)

        # Calculate 'xyz' tensor element of SHG spectra
        get_shg(nlodata, freqs=freqs, eta=0.025, pol='xyz',
                out_name=f'shg_xyz{tag}.npy')
        world.barrier()

        # Load the calculated SHG spectra (in units of nm/V)
        shg_xyz[spinpol] = np.load(f'shg_xyz{tag}.npy')[1] * 1e9
        assert shg_xyz[spinpol] == pytest.approx(shg_values, abs=1e-3), \
            np.max(np.abs(shg_xyz[spinpol] - shg_values))

    # import matplotlib.pyplot as plt
    # plt.plot(freqs, shg_xyz['spinpaired'])
    # plt.plot(freqs, shg_xyz['spinpol'])
    # plt.show()

    # Assert that the difference between spectra from spinpaired and
    # spinpolarised calculations is small

    # Absolute error
    shg_xyz_diff = shg_xyz['spinpaired'] - shg_xyz['spinpol']
    assert shg_xyz_diff.real == pytest.approx(0, abs=5e-4)
    assert shg_xyz_diff.imag == pytest.approx(0, abs=5e-4)

    # Relative error
    shg_xyz_avg = (shg_xyz['spinpaired'] + shg_xyz['spinpol']) / 2
    shg_xyz_rerr_real = shg_xyz_diff.real / shg_xyz_avg.real
    shg_xyz_rerr_imag = shg_xyz_diff.imag / shg_xyz_avg.imag
    assert shg_xyz_rerr_real == pytest.approx(0, abs=2e-3), \
        np.max(np.abs(shg_xyz_rerr_real))
    assert shg_xyz_rerr_imag == pytest.approx(0, abs=2e-3), \
        np.max(np.abs(shg_xyz_rerr_imag))
