"""Test the site spin splitting sum rules."""

import numpy as np
import pytest


def test():
    # Load data
    rc_r = np.load('rc_r.npy')
    dxc_r = np.load('dxc_r.npy')
    sp_dxc_r = np.load('sp_dxc_r.npy')
    dxc_nr = np.load('dxc_nr.npy')

    # Test the single-particle sum rule
    assert sp_dxc_r == pytest.approx(dxc_r, abs=0.01)

    # Test the two-particle sum rule
    # First, check that all values are positive and real
    assert np.all(dxc_nr.real > 0.)
    assert dxc_nr.imag == pytest.approx(0.)
    tp_dxc_nr = dxc_nr.real
    # Secondly, test accuracy for r > 1.1 Å, excluding unocc∈{0, 4, 8}
    rmask = rc_r > 1.1
    for tp_dxc_r in tp_dxc_nr[3:]:
        assert tp_dxc_r[rmask] == pytest.approx(dxc_r[rmask], abs=0.05)
    # Finally, test that there exist a hierachy of site spin splitting values,
    # when increasing the number of bands for 0.2 Å < r < 0.8 Å
    rmask = np.logical_and(0.2 < rc_r, rc_r < 0.8)
    for n in range(1, tp_dxc_nr.shape[0]):
        assert np.all(tp_dxc_nr[n - 1, rmask] < tp_dxc_nr[n, rmask])
    assert np.all(tp_dxc_nr[-1, rmask] < dxc_r[rmask])


if __name__ == '__main__':
    test()
