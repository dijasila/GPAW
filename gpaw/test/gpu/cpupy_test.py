from gpaw.gpu import cupy as cp
import pytest
import numpy as np


def test_basics():
    a = cp.empty(2)
    b = a.get()
    assert isinstance(b, np.ndarray)
    assert b.dtype == float


@pytest.mark.xfail
def test_grr():
    a = cp.empty((2, 2))
    a[:] = 2.5
    b = np.float64(3.0) * a
    assert isinstance(b, cp.ndarray)
