import pytest
import numpy as np
from gpaw.response.MPAsamp import mpa_frequency_sampling

from ase.units import Hartree as Ha

@pytest.mark.response
def test_mpa_samp():

    # print("npol=1, ps='1l', w1=0.1j, w2=1j, alpha=1:")
    w_grid = mpa_frequency_sampling(1, [complex(0,0.1), complex(0,1)], [0.1,0.1],
                                    ps='1l', alpha=1)
    assert w_grid == pytest.approx([0.+0.1j, 0.+1.j ])

    # print("npol=1, ps='2l', w1=0.1j, w2=1j, alpha=0:")
    w_grid = mpa_frequency_sampling(1, [complex(0,0.1), complex(0,1)], [0.1,0.1],
                                    ps='2l', alpha=0)
    assert w_grid == pytest.approx([0.+0.1j, 0.+1.j ])

    # print("npol=6, ps='2l', w1=0+1j, w2=2+1j, alpha=1:")
    w_grid = mpa_frequency_sampling(6, [complex(0,1), complex(2,1)], [0.01,0.1])
    assert w_grid == pytest.approx([0. + 0.01j, 0.25 + 0.1j, 0.5 + 0.1j, 
                                    1. + 0.1j, 1.5 + 0.1j, 2. + 0.1j, 0. + 1.j,
                                    0.25 + 1.j, 0.5 + 1.j, 1. + 1.j, 1.5 + 1.j,
                                    2.  + 1.j])