import pytest
import numpy as np

from gpaw.mpi import world
from gpaw.nlopt.basic import NLOData
from gpaw.nlopt.linear import get_chi_tensor


def test_chi_spinpol(mme_files):
    chi_values = np.array([-1690.61874551 + 74.98232115j,
                           -1675.75173888 + 73.98798631j,
                           -1661.08116901 + 73.01127033j,
                           -1646.60356913 + 72.05178209j,
                           -1632.31554910 + 71.10914090j,
                           -1618.21379336 + 70.18297621j,
                           -1604.29505895 + 69.27292726j,
                           -1590.55617363 + 68.37864280j,
                           -1576.99403400 + 67.49978077j,
                           -1563.60560369 + 66.63600805j,
                           -1550.38791166 + 65.78700017j])

    freqs = np.linspace(2.3, 2.4, 11)
    chi_xx = {}
    for spinpol in ['spinpaired', 'spinpol']:
        tag = '_spinpol' if spinpol == 'spinpol' else ''

        # Get pre-calculated nlodata from SiC fixtures
        nlodata = NLOData.load(mme_files[f'sic_pw{tag}'], comm=world)

        # Calculate tensor elements of susceptibility spectra
        get_chi_tensor(nlodata, freqs=freqs,
                       eta=0.05, out_name=f'chi{tag}.npy')
        world.barrier()

        # Load the calculated susceptibility
        chi_xx[spinpol] = np.load(f'chi{tag}.npy')[1]

        assert chi_xx[spinpol] == pytest.approx(chi_values, abs=5e-2)

    # Assert that the difference between spectra from spinpaired and
    # spinpolarised calculations is small

    # Absolute error
    assert chi_xx['spinpol'].real == pytest.approx(
        chi_xx['spinpaired'].real, abs=2e-2)

    # Relative error
    assert chi_xx['spinpol'].real == pytest.approx(
        chi_xx['spinpaired'].real, rel=1e-3)
    assert chi_xx['spinpol'].imag == pytest.approx(
        chi_xx['spinpaired'].imag, abs=1e-4)
