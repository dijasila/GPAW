import os

import pytest
from _pytest.tmpdir import _mk_tmp
from ase import Atoms
from ase.utils import devnull
from ase.build import bulk

from gpaw import GPAW
from gpaw.cli.info import info
from gpaw.mpi import world, broadcast


@pytest.fixture
def in_tmp_dir(request, tmp_path_factory):
    if world.rank == 0:
        path = _mk_tmp(request, tmp_path_factory)
    else:
        path = None
    path = broadcast(path)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='session')
def gpw_files(request, tmp_path_factory):
    if world.rank == 0:
        path = _mk_tmp(request, tmp_path_factory)
    else:
        path = None
    path = broadcast(path)
    return GPWFiles(path)


class GPWFiles:
    """Create gpw-files."""
    def __init__(self, path):
        self.path = path
        self.gpw_files = {}

    def __getitem__(self, name):
        if name not in self.gpw_files:
            print('gpw_files fixture:', name)
            rawname, _, _ = name.partition('_wfs')
            calc = getattr(self, rawname)()
            path = self.path / (rawname + '.gpw')
            calc.write(path)
            self.gpw_files[rawname] = path
            path = self.path / (rawname + '_wfs.gpw')
            calc.write(path, mode='all')
            self.gpw_files[rawname + '_wfs'] = path
        return self.gpw_files[name]

    def bcc_li_pw(self):
        return self.bcc_li({'name': 'pw', 'ecut': 200})

    def bcc_li_fd(self):
        return self.bcc_li({'name': 'fd'})

    def bcc_li_lcao(self):
        return self.bcc_li({'name': 'lcao'})

    def bcc_li(self, mode):
        li = bulk('Li', 'bcc', 3.49)
        li.calc = GPAW(mode=mode,
                       kpts=(3, 3, 3),
                       txt=self.path / f'bcc_li_{mode["name"]}.txt')
        li.get_potential_energy()
        return li.calc

    def h2_pw(self):
        return self.h2({'name': 'pw', 'ecut': 200})

    def h2_fd(self):
        return self.h2({'name': 'fd'})

    def h2_lcao(self):
        return self.h2({'name': 'lcao'})

    def h2(self, mode):
        h2 = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]])
        h2.center(vacuum=2.5)
        h2.calc = GPAW(mode=mode,
                       txt=self.path / f'h2_{mode["name"]}.txt')
        h2.get_potential_energy()
        return h2.calc


class GPAWPlugin:
    def __init__(self):
        if world.rank == 0:
            print()
            info()

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        from gpaw.mpi import size
        terminalreporter.section('GPAW-MPI stuff')
        terminalreporter.write(f'size: {size}\n')


def pytest_configure(config):
    if world.rank != 0:
        try:
            tw = config.get_terminal_writer()
        except AttributeError:
            pass
        else:
            tw._file = devnull
    config.pluginmanager.register(GPAWPlugin(), 'pytest_gpaw')


def pytest_runtest_setup(item):
    """Skip some tests.

    If:

    * they depend on libxc and GPAW is not compiled with libxc
    * they are before $PYTEST_START_AFTER
    """
    from gpaw import libraries

    if world.size > 1:
        for mark in item.iter_markers():
            if mark.name == 'serial':
                pytest.skip('Only run in serial')

    if item.location[0] <= os.environ.get('PYTEST_START_AFTER', ''):
        pytest.skip('Not after $PYTEST_START_AFTER')
        return

    if libraries['libxc']:
        return

    if any(mark.name in {'libxc', 'mgga'}
           for mark in item.iter_markers()):
        pytest.skip('No LibXC.')
