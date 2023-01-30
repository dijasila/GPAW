from gpaw.gpu import backends

backend = backends.HostBackend()
array = backend.array


def setup(**kwargs):
    global backend
    global array

    debug = kwargs.pop('debug', False)
    if debug == 'sync':
        debug = True
        debug_sync = True
    else:
        debug_sync = False

    args = {
        'debug': debug,
        'debug_sync': debug_sync,
    }

    if kwargs.pop('cuda', False):
        from gpaw.gpu.cuda import CUDA
        backend = CUDA(**args)
    else:
        backend = backends.HostBackend(**args)
    array = backend.array

    for key in kwargs:
        print(f'Unknown GPU parameter: {key}')


def init(rank=0):
    global backend

    backend.init(rank)

    if backend.debug:
        import platform
        print('[{0}] GPU device {1} initialised (on host {2}).'.format(
            rank, backend.device_no, platform.node()))
    return True


__all__ = ['arrays', 'backends', 'parameters', 'cuda']


# for ease of use, make the module behave as if it would be the backend,
#   i.e. gpu.enabled == gpu.backend.enabled etc.
def __getattr__(key):
    if key in __all__:
        import importlib
        return importlib.import_module("." + key, __name__)
    elif key in backend.api:
        return getattr(backend, key)
    else:
        raise AttributeError(f"module {__name__} has no attribute {key}")
