import numpy as np

from gpaw.gaunt import gaunt


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


def lm_indices(L):
    l = int(np.sqrt(L))
    m = L - l * (l + 1)
    return l, m
