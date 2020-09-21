"""Test point-group CLI."""
import pytest
from ase.io.cube import write_cube

from gpaw import GPAW
from gpaw.point_groups.cli import main


@pytest.mark.serial
def test_pg_cli(gpw_files, capsys):
    file = gpw_files['h2o_lcao_wfs']
    main(f'C2v {file} -c O -b 0:4'.split())
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert lines[1][1] == 'Yes'
    assert '-'.join(lines[n][4] for n in range(3, 7)) == 'A1-B2-A1-B1'


@pytest.mark.serial
def test_pg_cli_cube(gpw_files, capsys, in_tmp_dir):
    calc = GPAW(gpw_files['h2o_lcao_wfs'])
    wf = calc.get_pseudo_wave_function(0)
    cube = in_tmp_dir / 'h2o.cube'
    with cube.open('w') as fd:
        write_cube(fd, calc.atoms, wf)
    main(f'C2v {cube} -c O'.split())
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert lines[1][1] == 'Yes'
    assert lines[3][4] == 'A1'
