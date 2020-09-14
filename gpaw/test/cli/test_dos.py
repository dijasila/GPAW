"""Test CLI-dos command."""
import pytest
from gpaw.cli.main import main


@pytest.mark.serial
@pytest.mark.parametrize(
    'options',
    ['',
     '-t',
     '-i',
     '-a Li-sp',
     '-w 0',
     '-r -5 5',
     '-n 50',
     '--soc'])
def test_dos(bcc_li_gpw, options):
    args = ('dos ' + bcc_li_gpw + ' ' + options).split()
    main(args)
