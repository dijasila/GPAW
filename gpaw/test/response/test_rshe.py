"""Test functionality to expand a function on a spherical grid in real
spherical harmonics."""

import pytest
import itertools

import numpy as np


def generate_combinations():
    """Generate combinations of one and two spherical harmonic indices.
    """
    lmax = 5
    nL = (lmax + 1)**2
    L_L = np.arange(nL)
    twoL_i = list(itertools.combinations(L_L, 2))

    Lcomb_i = list(L_L) + twoL_i

    return Lcomb_i


@pytest.mark.response
@pytest.mark.parametrize('Lcomb', generate_combinations())
def test_rshe(Lcomb):
    pass
