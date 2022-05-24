"""Test gpaw symmetry command."""
import pytest
from gpaw.cli.main import main

result = """\
Symmetries present (total): 48
Symmetries with fractional translations: 0
512 k-points: 8 x 8 x 8 Monkhorst-Pack grid + [1/16,1/16,1/16]
29 k-points in the irreducible part of the Brillouin zone
"""


@pytest.mark.serial
def test_symmetry(gpw_files, capsys):
    args = ['symmetry',
            str(gpw_files['bcc_li_pw']),
            '-k',
            '{density:3,gamma:1}']
    main(args)
    out = capsys.readouterr().out
    print(out)
    assert out == result
