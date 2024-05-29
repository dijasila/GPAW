import pytest
from ase.io.ulm import ulmopen
import ase.dft.bandgap


def test_yaml(gpw_files):
    yaml = pytest.importorskip('yaml')
    gpw = gpw_files['h2_pw']
    if ulmopen(gpw).version < 4:
        pytest.skip('Old gpw-file')
    if not hasattr(ase.dft.bandgap, 'GapInfo'):
        pytest.skip('ASE too old')
    txt = gpw.with_name('h2_pw.txt')
    with txt.open() as fd:
        header, body = yaml.safe_load_all(fd)
    assert body['Gap'] == pytest.approx(11.296, abs=0.001)
