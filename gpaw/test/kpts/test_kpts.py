from gpaw.kpt_descriptor import KPointDescriptor


def test_almost_gamma():
    kd = KPointDescriptor([[1e-15, 0, 0]])
    assert kd.gamma
    assert not kd.ibzk_kc.any()
