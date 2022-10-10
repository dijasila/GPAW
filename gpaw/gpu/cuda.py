import atexit

import _gpaw
from gpaw.gpu.backends import BaseBackend

class CUDA(BaseBackend):
    # from pycuda import driver as _driver
    # from gpaw.gpu.arrays import PyCudaArrayInterface
    from cupy.cuda import runtime, get_current_stream
    from gpaw.gpu.arrays import CuPyArrayInterface

    label = 'cuda'
    enabled = True
    array = CuPyArrayInterface()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        #_gpaw.set_gpaw_cuda_debug(self.debug)  Commented out not to have the C-interface

    def init(self, rank=0):
        if self.device_ctx is not None:
            return
        atexit.register(self.delete)

        # initialise CUDA driver
        # self._driver.init()

        # select device (round-robin based on MPI rank)
        self.device_no = (rank) % self.runtime.getDeviceCount() 

        # create and activate CUDA context
        # device = self._driver.Device(self.device_no)
        # self.device_ctx = device.make_context(
        #         flags=self._driver.ctx_flags.SCHED_YIELD)
        # self.device_ctx.push()
        # self.device_ctx.set_cache_config(self._driver.func_cache.PREFER_L1)

        # initialise C parameters and memory buffers
        _gpaw.gpaw_cuda_setdevice(self.device_no)
        _gpaw.gpaw_cuda_init()

    def delete(self):
        if self.device_ctx is not None:
            # deallocate memory buffers
            _gpaw.gpaw_cuda_delete()
            # deactivate and destroy CUDA context
            # self.device_ctx.pop()
            # self.device_ctx.detach()
            # del self.device_ctx
            # self.device_ctx = None

    def copy_to_host(self, src, tgt=None, stream=None):
        return self.array.copy_to_host(src, tgt=tgt, stream=stream)

    def copy_to_device(self, src, tgt=None, stream=None):
        return self.array.copy_to_device(src, tgt=tgt, stream=stream)

    def is_device_array(self, x):
        return isinstance(x, self.array.Array)

    def memcpy_dtod(self, tgt, src, n):
        self.array.memcpy_dtod(tgt, src)

    def synchronize(self):
        self.get_current_stream().synchronize() 
