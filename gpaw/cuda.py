import numpy as np
import warnings
try:
    import pycuda.driver as drv
    import pycuda.tools as tools
except ImportError:
    pass

import gpaw.gpuarray as gpuarray
import _gpaw


debug = False
debug_sync = False
cuda_ctx = None

class DebugCudaError(Exception):
    pass

class DebugCudaWarning(UserWarning):
    pass



def init(rank=0):
    """
    """
    
    global cuda_ctx

    if cuda_ctx is not None:
        return True

    try:
        drv.init()
    except NameError:
        errmsg = "PyCUDA not found."
        raise NameError(errmsg)
        return False

    devno=(rank+1) % drv.Device.count()
    
    if drv.Device(devno).get_attribute(drv.device_attribute.COMPUTE_MODE) is drv.compute_mode.EXCLUSIVE:
        cuda_ctx=tools.make_default_context()
    else:
        current_dev = drv.Device(devno)
        cuda_ctx = current_dev.make_context()
    cuda_ctx.push()                
        
    cuda_ctx.set_cache_config(drv.func_cache.PREFER_L1)
    _gpaw.cuda_init()
    return True
    
def delete():
    """
    """
    global cuda_ctx
    
    if cuda_ctx is not None:
        _gpaw.cuda_delete()
        cuda_ctx.pop() #deactivate again
        cuda_ctx.detach() #delete it

    del cuda_ctx
    cuda_ctx = None

def get_context():
    """
    """
    global cuda_ctx
    return cuda_ctx

def debug_test(x,y,text,reltol=1e-12,abstol=1e-13,raise_error=False):
    """
    """

    if isinstance(x,gpuarray.GPUArray):
        x_cpu=x.get()
        x_type='GPU'
    else:
        x_cpu=x
        x_type='CPU'

    if isinstance(y,gpuarray.GPUArray):
        y_cpu=y.get()
        y_type='GPU'
    else:
        y_cpu=y
        y_type='CPU'
    
    if not  np.allclose(x_cpu,y_cpu,reltol,abstol):
        diff=abs(y_cpu-x_cpu)
        if isinstance(diff, (float, complex)):
            warnings.warn('%s error %s %s %s %s diff: %s' \
                          % (text,y_type,y_cpu,x_type,x_cpu,abs(y_cpu-x_cpu)),  \
                          DebugCudaWarning,stacklevel=2)
        else:
            error_i=np.unravel_index(np.argmax(diff - reltol * abs(y_cpu)), \
                                     diff.shape)
            warnings.warn('%s max rel error pos: %s %s: %s %s: %s diff: %s' \
                          % (text,error_i,y_type,y_cpu[error_i], \
                             x_type,x_cpu[error_i], \
                             abs(y_cpu[error_i]-x_cpu[error_i])),  \
                          DebugCudaWarning,stacklevel=2)
            error_i=np.unravel_index(np.argmax(diff),diff.shape)
            warnings.warn('%s max abs error pos: %s %s: %s %s: %s diff:%s' \
                          % (text,error_i,y_type,y_cpu[error_i], \
                             x_type,x_cpu[error_i], \
                             abs(y_cpu[error_i]-x_cpu[error_i])),  \
                          DebugCudaWarning, stacklevel=2)
            warnings.warn('%s error shape: %s dtype: %s' \
                          % (text,x_cpu.shape,x_cpu.dtype),  \
                          DebugCudaWarning, stacklevel=2)

            warnings.warn('%s error: %s %s' \
                          % (text,x_type,x_cpu),  \
                          DebugCudaWarning, stacklevel=2)
            warnings.warn('%s error: %s %s' \
                          % (text,y_type,y_cpu),  \
                          DebugCudaWarning, stacklevel=2)
        
        if raise_error:
            raise DebugCudaError
        return False
    
    return True
