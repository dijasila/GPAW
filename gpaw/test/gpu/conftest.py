import pytest


@pytest.fixture(scope='session')
def gpu():
    from gpaw import gpu
    try:
        gpu.setup()
    except ImportError as err:
        pytest.skip(reason=f'Cannot import GPU backend ({err})')
    except Exception as err:
        pytest.skip(reason=f'Cannot find GPU devices ({err})')

    return gpu
