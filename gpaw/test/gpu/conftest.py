import pytest


@pytest.fixture(scope='session', autouse=True)
def gpu():
    cupy = pytest.importorskip('cupy')
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip(reason=f'Cannot find GPU devices: {err}')

    from gpaw import gpu
    gpu.setup(cuda=True)
    gpu.init()
    return gpu
