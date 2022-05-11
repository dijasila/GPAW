import pytest
from ase.io import read


def test_yaml(in_tmp_dir):
    pytest.importorskip('yaml')
    (in_tmp_dir / 'y.yaml').write_text('# gpaw2022  ')
    read('y.yaml')
