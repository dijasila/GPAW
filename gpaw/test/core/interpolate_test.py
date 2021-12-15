import pytest
from gpaw.core import UniformGrid


@pytest.mark.ci
def test_fft_interpolation():
    a = UniformGrid(cell=[1, 1, 1], size=(4, 4, 4)).zeros()
    b = UniformGrid(cell=[1, 1, 1], size=(8, 8, 8)).zeros()
    a.data[2, 2, 2] = 1.0
    a.fft_interpolate(out=b)
    assert (b.data[::2, ::2, ::2] == a.data).all()

    b.fft_restrict(out=a)
    assert a.integrate() == b.integrate()
