import platform

from .backend import DummyBackend

backend = DummyBackend()

def setup(**kwargs):
    global backend

    debug = kwargs.pop('debug', False)
    if debug == 'sync':
        debug = True
        debug_sync = True
    else:
        debug_sync = False
    use_hybrid_blas = bool(kwargs.pop('hybrid_blas', False))

    args = {
            'debug': debug,
            'debug_sync': debug_sync,
            'use_hybrid_blas': use_hybrid_blas,
            }

    if kwargs.pop('cuda', False):
        from .cuda import CUDA
        backend = CUDA(**args)
    else:
        backend = DummyBackend(**args)

    for key in kwargs:
        print(f'Unknown GPU parameter: {key}')


def init(rank=0):
    global backend

    try:
        backend.init(rank)
    except Exception:
        raise Exception("GPU could not be initialised")

    if backend.debug:
        print('[{0}] GPU device {1} initialised (on host {2}).'.format(
            rank, backend.device_no, platform.node()))
    return True


# for ease of use, make the module behave as if it would be the backend,
#   i.e. gpu.enabled == gpu.backend.enabled etc.
def __getattr__(key):
    return getattr(backend, key)
