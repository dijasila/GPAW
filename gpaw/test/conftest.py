import os

import pytest
from ase.utils import devnull

from gpaw.mpi import world


@pytest.fixture
def in_tmp_dir(tmpdir):
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        yield tmpdir
    finally:
        os.chdir(cwd)


class GPAWPlugin:
    def __init__(self):
        print('hello')

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        from gpaw.mpi import rank, size
        terminalreporter.section('GPAW-MPI stuff')
        terminalreporter.write(f'rank, size: {rank}, {size}')

    def pytest_report_header(self, config, startdir):
        return 'hej'

    def pytest_report_collectionfinish(self, config, startdir, items):
        tw = config.get_terminal_writer()
        if world.rank != 0:
            tw._file = devnull

    # def pytest_make_collect_report(self, collector):
    #     pass


def pytest_configure(config):
    config.pluginmanager.register(GPAWPlugin(), 'pytest_gpaw')
    # print(dir(config))


def pytest_sessionstart(session):
    print(dir(session))
    szdagljkh

