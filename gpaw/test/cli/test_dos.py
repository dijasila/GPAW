import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.cli.main import main
from gpaw.mpi import size


@pytest.fixture
def gpw(tmp_path):
    li = bulk('Li', 'bcc', 3.49)
    li.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                   kpts=(3, 3, 3))
    li.get_potential_energy()
    li.calc.write(tmp_path / 'li.gpw')
    return str(tmp_path / 'li.gpw')


@pytest.mark.skipif(size > 1, reason='Not serial')
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
