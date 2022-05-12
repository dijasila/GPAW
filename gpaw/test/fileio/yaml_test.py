import pytest
from ase.io import read


def test_yaml(gpw_files):
    pytest.importorskip('yaml')
    a = read(gpw_files['h2_pw'].with_name('h2_pw.txt'))
    print(a)
