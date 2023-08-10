import pytest

import numpy as np
from itertools import permutations

from gpaw.spherical_harmonics import Y
from gpaw.gaunt import gaunt, super_gaunt


def test_contraction_rule(lmax: int = 3):
    """Test that two spherical harmonics can be contracted to one."""
    G_LLL = gaunt(lmax)
    L1max, L2max, L3max = G_LLL.shape
    assert L2max == L3max

    x, y, z = unit_sphere_test_coordinates()
    for L1 in range(L1max):
        # In order to include all finite Gaunt coefficients in the expansion,
        # l1 + l2 has to be within the range of available L3 coefficients.
        l1 = int(np.sqrt(L1))
        l3max = int(np.sqrt(L3max)) - 1
        l2max = l3max - l1
        L2max = (l2max + 1)**2
        for L2 in range(L2max):
            product = Y(L1, x, y, z) * Y(L2, x, y, z)
            expansion = 0.
            for L3 in range(L3max):
                expansion += G_LLL[L1, L2, L3] * Y(L3, x, y, z)
            assert expansion == pytest.approx(product)


def test_selection_rules(lmax: int = 3):
    """Test Gaunt coefficient selection rules."""
    G_LLL = gaunt(lmax)
    L1max, L2max, L3max = G_LLL.shape

    for L1 in range(L1max):
        # Convert composit index to (l,m)
        l1, m1 = lm_indices(L1)
        for L2 in range(L2max):
            l2, m2 = lm_indices(L2)
            for L3 in range(L3max):
                l3, m3 = lm_indices(L3)
                # In order for the Gaunt coefficients to be finite, the
                # following conditions should be met
                if abs(G_LLL[L1, L2, L3]) > 1e-8:
                    assert (l1 + l2 + l3) % 2 == 0, \
                        f'l1 + l2 + l3 = {l1+l2+l3} should be an even integer'
                    assert abs(l1 - l2) <= l3 <= l1 + l2, f'{l1, l2, l3}'
                    assert m1 + m2 + m3 == 0 or \
                        m1 + m2 - m3 == 0 or \
                        m3 + m1 - m2 == 0 or \
                        m2 + m3 - m1 == 0, f'{m1, m2, m3}'


def test_permutation_symmetry(lmax: int = 3):
    """Test that the Gaunt coefficients are permutationally invariant"""
    G_LLL = gaunt(lmax)
    L1max, L2max, L3max = G_LLL.shape
    assert L2max == L3max

    for L1 in range(L1max):
        # Permutations between all indices
        for L2 in range(L1max):
            for L3 in range(L1max):
                for Lp1, Lp2, Lp3 in permutations([L1, L2, L3]):
                    abs(G_LLL[L1, L2, L3] - G_LLL[Lp1, Lp2, Lp3]) < 1e-10
        # Permutations between L2 and L3
        for L2 in range(L2max):
            for L3 in range(L3max):
                abs(G_LLL[L1, L2, L3] - G_LLL[L1, L3, L2]) < 1e-10


def test_super_contraction_rule(lmax: int = 2):
    """Test that three spherical harmonics can be contracted to one."""
    G_LLLL = super_gaunt(lmax)
    L1max, L2max, L3max, L4max = G_LLLL.shape
    assert L1max == L2max

    x, y, z = unit_sphere_test_coordinates()
    for L1 in range(L1max):
        for L2 in range(L2max):
            for L3 in range(L3max):
                product = Y(L1, x, y, z) * Y(L2, x, y, z) * Y(L3, x, y, z)
                expansion = 0.
                for L4 in range(L4max):
                    expansion += G_LLLL[L1, L2, L3, L4] * Y(L4, x, y, z)
                assert expansion == pytest.approx(product)
        

def unit_sphere_test_coordinates():
    """Unit-sphere coordinates to test"""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 11),
                             np.linspace(0, 2 * np.pi, 21),
                             indexing='ij')
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def lm_indices(L):
    l = int(np.sqrt(L))
    m = L - l * (l + 1)
    return l, m
