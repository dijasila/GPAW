import numpy as np
import pytest
from ase.data import chemical_symbols

from gpaw.response.paw import calculate_pair_density_correction
from gpaw.setup import create_setup


def pawdata():
    for symbol in chemical_symbols:
        try:
            setup = create_setup(symbol)
        except FileNotFoundError:
            pass
        else:
            yield setup


@pytest.mark.serial
@pytest.mark.parametrize('pawdata', pawdata())
def test_paw_corrections(pawdata):
    G_Gv = np.zeros((5, 3))
    G_Gv[:, 0] = np.linspace(0, 20, 5)
    calculate_pair_density_correction(G_Gv, pawdata=pawdata)
