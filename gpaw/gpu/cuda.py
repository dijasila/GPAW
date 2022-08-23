import atexit

import _gpaw

from .backend import DummyBackend

class CUDA(DummyBackend):
    from pycuda import driver as _driver
    from pycuda.driver import memcpy_dtod

    enabled = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _gpaw.set_gpaw_cuda_debug(self.debug)

    def init(self, rank=0):
        if self.device_ctx is not None:
            return
        atexit.register(self.delete)

        # initialise CUDA driver
        self._driver.init()

        # select device (round-robin based on MPI rank)
        self.device_no = (rank) % self._driver.Device.count()

        # create and activate CUDA context
        device = self._driver.Device(self.device_no)
        self.device_ctx = device.make_context(
                flags=self._driver.ctx_flags.SCHED_YIELD)
        self.device_ctx.push()
        self.device_ctx.set_cache_config(self._driver.func_cache.PREFER_L1)

        # initialise C parameters and memory buffers
        _gpaw.gpaw_cuda_setdevice(self.device_no)
        _gpaw.gpaw_cuda_init()

    def delete(self):
        if self.device_ctx is not None:
            # deallocate memory buffers
            _gpaw.gpaw_cuda_delete()
            # deactivate and destroy CUDA context
            self.device_ctx.pop()
            self.device_ctx.detach()
            del self.device_ctx
            self.device_ctx = None

    def get_context(self):
        return self.device_ctx

    def debug_test(self, x, y, text, reltol=1e-12, abstol=1e-13,
                   raise_error=False):
        import warnings
        import numpy as np

        from gpaw import gpuarray

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
