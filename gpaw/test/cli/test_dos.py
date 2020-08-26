"""Test CLI-dos command."""
import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.cli.main import main
from gpaw.mpi import size


@pytest.fixture(scope='module')
def gpw(tmp_path_factory):
    """Create gpw-file."""
    li = bulk('Li', 'bcc', 3.49)
    li.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                   kpts=(3, 3, 3))
    li.get_potential_energy()
    path = tmp_path_factory.mktemp('gpw-files') / 'li.gpw'
    li.calc.write(path)
    return str(path)


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
def test_dos(gpw, options):
    args = ('dos ' + gpw + ' ' + options).split()
    main(args)
