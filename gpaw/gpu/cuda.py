import atexit

import _gpaw

from .backend import BaseBackend

class CUDA(BaseBackend):
    from pycuda import driver as _driver

    from gpaw.gpu import pycuda as _gpuarray

    label = 'cuda'
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

    def copy_to_host(self, x):
        return x.get()

    def copy_to_device(self, x):
        return self._gpuarray.to_gpu(x)

    def is_device_array(self, x):
        return isinstance(x, self._gpuarray.GPUArray)

    def memcpy_dtod(self, tgt, src, n):
        self._driver.memcpy_dtod(tgt.gpudata, src.gpudata, n)
