import os

import pytest


@pytest.fixture
def in_tmp_dir(tmpdir):
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        yield tmpdir
    finally:
        os.chdir(cwd)


class GPAWPlugin:
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        from gpaw.mpi import rank, size
        terminalreporter.section('GPAW-MPI stuff')

        terminalreporter.write(f'rank, size: {rank}, {size}')
        print(dir(terminalreporter))

    def pytest_report_header(self, config, startdir):
        return 'hej'

    def pytest_report_collectionfinish(self, config, startdir, items):
        return '******' * 9


def pytest_configure(config):
    config.pluginmanager.register(GPAWPlugin(), 'pytest_gpaw')
