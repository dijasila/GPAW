import pytest
from gpaw.mpi import world


@pytest.fixture(scope='session')
def gpu():
    from gpaw import gpu
    try:
        gpu.old_setup(enabled=True)
        gpu.old_init(world.rank)
    except ImportError as err:
        pytest.skip(reason=f'Cannot import GPU backend ({err})')
    except Exception as err:
        pytest.skip(reason=f'Cannot find GPU devices ({err})')

    return gpu
