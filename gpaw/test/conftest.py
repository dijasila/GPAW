import os

import pytest
from _pytest.tmpdir import _mk_tmp
from _pytest.config import hookimpl
from ase.utils import devnull

from gpaw.cli.info import info
from gpaw.test import mpi_size
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

@hookimpl(tryfirst=True)
def pytest_runtest_call(item) -> None:
    if mpi_size == 1:
        item.runtest()
        return
    # for n in dir(item):
    #     print(n, getattr(item, n))

    mod = item.module.__name__
    name = item.obj.__name__

    import subprocess
    cmd = ['mpiexec', '-n', '2', 'python3', '-c',
           f'from {mod} import {name}; {name}(42)']
    print(cmd)
    return True
    r = subprocess.run(cmd)
    print(r, dir(r), r.stdout, r.stderr)
    return True

def pytest_runtest_setup(item):
    """Skip tests that depend on libxc if not compiled with libxc."""
    from gpaw import libraries
    if mpi_size == 1 and any(mark.name == 'parallel'
                             for mark in item.iter_markers()):
        pytest.skip('Not parllel.')
    if libraries['libxc']:
        return
    if any(mark.name in {'libxc', 'mgga'}
           for mark in item.iter_markers()):
        pytest.skip('No LibXC.')


