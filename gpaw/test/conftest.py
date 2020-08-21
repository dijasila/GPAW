import os

import pytest
from _pytest.tmpdir import _mk_tmp
from ase.utils import devnull

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

    if item.location[0] <= os.environ.get('PYTEST_START_AFTER', ''):
        pytest.skip('Not after $PYTEST_START_AFTER')
        return

    if libraries['libxc']:
        return

    if any(mark.name in {'libxc', 'mgga'}
           for mark in item.iter_markers()):
        pytest.skip('No LibXC.')
