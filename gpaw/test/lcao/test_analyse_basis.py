from subprocess import check_output
from pathlib import Path


def test_analyse_basis(in_tmp_dir):
    argv = ['gpaw-analyse-basis', 'H.dzp.basis', '--save-figs']
    out = check_output(argv, encoding='utf-8')
    assert '1s-sz' in out
    assert '1s-dz' in out
    assert 'p-type' in out
    assert Path('H.dzp.png').is_file()
