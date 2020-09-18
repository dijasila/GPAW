"""Test point-group CLI."""


def test_pg_cli(gpw_files, capsys):
    from gpaw.point_groups.cli import main
    file = gpw_files['h2o_lcao_wfs']
    main(f'C2v {file} -c O -b 0:4'.split())
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert lines[1][1] == 'Yes'
    assert '-'.join(lines[n][4] for n in range(3, 7)) == 'A1-B2-A1-B1'
