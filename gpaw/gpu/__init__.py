from gpaw.gpu import backends

backend = backends.HostBackend()
array = backend.array


def setup(enabled=False):
    global backend
    global array

    if enabled:
        from gpaw.gpu.cuda import CUDA
        backend = CUDA()
    else:
        backend = backends.HostBackend()
    array = backend.array

    return backend


def init(rank=0):
    global backend

    backend.init(rank)


__all__ = ['arrays', 'backends', 'parameters', 'cuda']
