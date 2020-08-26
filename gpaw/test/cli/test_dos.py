import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.cli.main import main


@pytest.fixture
def gpw(in_tmp_dir):
    li = bulk('Li', 'bcc', 3.49)
    li.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                   kpts=(3, 3, 3))
    li.get_potential_energy()
    li.calc.write('li.gpw')
    return str(in_tmp_dir / 'li.gpw')


@pytest.mark.parametrize(
    'args',
    [[],
     ['-t'],
     ['-i'],
     ['-a', 'Li-sp'],
     ['-w', '0'],
     ['-r', '-5', '5'],
     ['-n', '50'],
     ['--soc']])
def test_dos(gpw, args):
    args = ['dos', gpw] + args
    main(args)
