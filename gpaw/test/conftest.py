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
            print('-' * 24, '-' * 50)
            info()
            print('-' * 24, '-' * 50)

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        from gpaw.mpi import rank, size
        terminalreporter.section('GPAW-MPI stuff')
        terminalreporter.write(f'rank, size: {rank}, {size}')


def pytest_configure(config):
    tw = config.get_terminal_writer()
    if world.rank != 0:
        tw._file = devnull
    config.pluginmanager.register(GPAWPlugin(), 'pytest_gpaw')
