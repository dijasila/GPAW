import numpy as np
import warnings
try:
    import pycuda.driver as drv
    import pycuda.tools as tools

except ImportError:
    pass

import gpaw.gpuarray as gpuarray


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
    try:
        drv.init()
    except NameError:
        errmsg = "PyCUDA not found."
        raise NameError(errmsg)
        return False
    
    if drv.Device(0).get_attribute(drv.device_attribute.COMPUTE_MODE) is drv.compute_mode.EXCLUSIVE:
        cuda_ctx=tools.make_default_context()
    else:
        current_dev = drv.Device(rank % drv.Device.count())
        cuda_ctx = current_dev.make_context()
    cuda_ctx.push()                
        
    cuda_ctx.set_cache_config(drv.func_cache.PREFER_L1)
    return True
    
def delete():
    """
    """
    if cuda_ctx is not None:
        cuda_ctx.pop() #deactivate again
        cuda_ctx.detach() #delete it
    

def get_context():
    """
    """
    return cuda_ctx

def debug_test(x,y,text,reltol=1e-12,abstol=1e-13,raise_error=False):
    """
    """

    if isinstance(x,gpuarray.GPUArray):
        x_cpu=x.get()
    else:
        x_cpu=x

    if isinstance(y,gpuarray.GPUArray):
        y_cpu=y.get()
    else:
        y_cpu=y
    
    if not  np.allclose(x_cpu,y_cpu,reltol,abstol):
        diff=abs(y_cpu-x_cpu)
        if isinstance(diff, (float, complex)):
            warnings.warn('%s error val: %s %s diff: %s' \
                          % (text,y_cpu,x_cpu,abs(y_cpu-x_cpu)),  \
                          DebugCudaWarning,stacklevel=2)
        else:
            error_i=np.unravel_index(np.argmax(diff - reltol * abs(y_cpu)), \
                                     diff.shape)
            warnings.warn('%s max rel error pos: %s val: %s %s diff: %s' \
                          % (text,error_i,y_cpu[error_i],x_cpu[error_i], \
                             abs(y_cpu[error_i]-x_cpu[error_i])),  \
                          DebugCudaWarning,stacklevel=2)
            error_i=np.unravel_index(np.argmax(diff),diff.shape)
            warnings.warn('%s max abs error pos: %s val: %s %s diff:%s' \
                          % (text,error_i,y_cpu[error_i],x_cpu[error_i], \
                             abs(y_cpu[error_i]-x_cpu[error_i])),  \
                          DebugCudaWarning, stacklevel=2)
        
        if raise_error:
            raise DebugCudaError
        return False
    
    return True
