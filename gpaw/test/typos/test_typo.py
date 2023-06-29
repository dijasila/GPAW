from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
import pytest


@pytest.mark.ci
def test_typo():
    assert 'xisting' not in DipoleMomentWriter.__doc__.split()
    