from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter


def test_typo():
    assert 'xisting' not in DipoleMomentWriter.__doc__.split()
    