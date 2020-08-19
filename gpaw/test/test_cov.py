import pytest
from gpaw.mpi import size


@pytest.mark.cov
def test_cov():
    print(1)
    if size == 2:
        print(2)
