from gpaw.gpu import backends

backend = backends.HostBackend()
array = backend.array


def setup(**kwargs):
    global backend
    global array

    args = {
        'debug': kwargs.pop('debug', False),
        'debug_sync': kwargs.pop('debug_sync', False),
    }

    if kwargs.pop('cuda', False):
        from gpaw.gpu.cuda import CUDA
        backend = CUDA(**args)
    else:
        backend = backends.HostBackend(**args)
    array = backend.array

    for key in kwargs:
        print(f'Unknown GPU parameter: {key}')

    return backend


def init(rank=0):
    global backend

    backend.init(rank)

    if backend.debug:
        import platform
        print('[{0}] GPU device {1} initialised (on host {2}).'.format(
            rank, backend.device_no, platform.node()))


__all__ = ['arrays', 'backends', 'parameters', 'cuda']
