"""Test hyperfine parameter CLI."""
import pytest
import numpy as np

from gpaw.hyperfine import main


@pytest.mark.serial
def test_hyperfine_cli(gpw_files, capsys):
    file = gpw_files['o2_pw']
    main(f'{file} -g O:-0.757 -u MHz'.split())
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    print(lines)
    for line in lines[3:5]:
        errors = (np.array([float(x) for x in line[2:]]) -
                  [0.689, -76.65, 93.45, -46.73, -46.73, 0.0, 0.0, 0.0])
        assert abs(errors).max() < 0.1
    assert float(lines[6][3]) == 2.0
    assert float(lines[9][1]) == -0.757
