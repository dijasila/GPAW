import platform
import atexit

import _gpaw

class CUDA:
    import pycuda.driver as drv
    import pycuda.tools as tools
    from pycuda.driver import memcpy_dtod
    from gpaw import gpuarray

    enabled = True
    debug = False
    debug_sync = False
    use_hybrid_blas = False
    device_no = None
    device_ctx = None

    def __init__(self, debug=False, debug_sync=False, use_hybrid_blas=False):
        self.debug = debug
        self.debug_sync = debug_sync
        self.use_hybrid_blas = use_hybrid_blas
        _gpaw.set_gpaw_cuda_debug(self.debug)

    def init(self, rank=0):
        if self.device_ctx is not None:
            return True
        atexit.register(self.delete)

        # initialise CUDA driver
        drv.init()

        # select device (round-robin based on MPI rank)
        self.device_no = (rank) % drv.Device.count()

        # create and activate CUDA context
        device = drv.Device(self.device_no)
        self.device_ctx = device.make_context(flags=drv.ctx_flags.SCHED_YIELD)
        self.device_ctx.push()
        self.device_ctx.set_cache_config(drv.func_cache.PREFER_L1)

        # initialise C parameters and memory buffers
        _gpaw.gpaw_cuda_setdevice(self.device_no)
        _gpaw.gpaw_cuda_init()
        if debug:
            print('[{0}] GPU device {1} initialised (on host {2}).'.format(
                rank, self.device_no, platform.node()))
        return True

    def delete():
        if self.cuda_ctx is not None:
            # deallocate memory buffers
            _gpaw.gpaw_cuda_delete()
            # deactivate and destroy CUDA context
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()
            del self.cuda_ctx
            self.cuda_ctx = None

    def get_context():
        return self.cuda_ctx

    def debug_test(x, y, text, reltol=1e-12, abstol=1e-13, raise_error=False):
        import warnings
        import numpy as np

        class DebugCudaError(Exception):
            pass

        class DebugCudaWarning(UserWarning):
            pass

        if isinstance(x, gpuarray.GPUArray):
            x_cpu = x.get()
            x_type = 'GPU'
        else:
            x_cpu = x
            x_type = 'CPU'

        if isinstance(y, gpuarray.GPUArray):
            y_cpu = y.get()
            y_type = 'GPU'
        else:
            y_cpu = y
            y_type = 'CPU'

        if not np.allclose(x_cpu, y_cpu, reltol, abstol):
            diff = abs(y_cpu - x_cpu)
            if isinstance(diff, (float, complex)):
                warnings.warn('%s error %s %s %s %s diff: %s' \
                              % (text, y_type, y_cpu, x_type, x_cpu, \
                                 abs(y_cpu - x_cpu)), \
                              DebugCudaWarning, stacklevel=2)
            else:
                error_i = np.unravel_index(np.argmax(diff - reltol * abs(y_cpu)), \
                                           diff.shape)
                warnings.warn('%s max rel error pos: %s %s: %s %s: %s diff: %s' \
                              % (text, error_i, y_type, y_cpu[error_i], \
                                 x_type, x_cpu[error_i], \
                                 abs(y_cpu[error_i] - x_cpu[error_i])),  \
                              DebugCudaWarning, stacklevel=2)
                error_i = np.unravel_index(np.argmax(diff), diff.shape)
                warnings.warn('%s max abs error pos: %s %s: %s %s: %s diff:%s' \
                              % (text, error_i, y_type, y_cpu[error_i], \
                                 x_type, x_cpu[error_i], \
                                 abs(y_cpu[error_i] - x_cpu[error_i])),  \
                              DebugCudaWarning, stacklevel=2)
                warnings.warn('%s error shape: %s dtype: %s' \
                              % (text, x_cpu.shape, x_cpu.dtype),  \
                              DebugCudaWarning, stacklevel=2)

            if raise_error:
                raise DebugCudaError
            return False

        return True
