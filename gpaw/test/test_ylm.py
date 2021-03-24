from gpaw.spherical_harmonics import second_derivatives as d2y


def test_second_derivatives():
    """Test that only l2=l1-1 and l2=l1+2 gives non-zero values for
    lesser and greater respectively."""
    for l1 in range(5):
        for l2 in range(7):
            Y = d2y(l1, l2=l2, kind='lesser')
            assert Y.any() == (l2 == l1 - 2)
            Y = d2y(l1, l2=l2, kind='greater')
            assert Y.any() == (l2 == l1 + 2)
