backend = None

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

    for key in kwargs:
        print(f'Unknown GPU parameter: {key}')


def init(rank=0):
    global backend

    try:
        backend.init(rank)
    except Exception:
        raise Exception("GPU could not be initialised")
    return True


get_context = backend.get_context
debug_test = backend.debug_test
