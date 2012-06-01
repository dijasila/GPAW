import numpy as np
import warnings
import pycuda.driver as cuda
import pycuda.tools as cuda_tools
from pycuda.driver import func_cache

debug_cuda = False
debug_cuda_sync = False
cuda_ctx = None

class DebugCudaError(Exception):
    pass

class DebugCudaWarning(UserWarning):
    pass



def init(rank=0):
    """
    """
    global cuda_ctx
    cuda.init()
    if cuda.Device(0).get_attribute(cuda.device_attribute.COMPUTE_MODE) is cuda.compute_mode.EXCLUSIVE:
        cuda_ctx=cuda_tools.make_default_context()
    else:
        current_dev = cuda.Device(rank % cuda.Device.count())
        cuda_ctx = current_dev.make_context()
    cuda_ctx.push()                
        
    cuda_ctx.set_cache_config(func_cache.PREFER_L1)
    return cuda_ctx
    
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

def debug_cuda_test(x,y,text,reltol=1e-12,abstol=1e-13,raise_error=False):
    """
    """
    if not  np.allclose(x,y,reltol,abstol):
        diff=abs(y-x)
        if isinstance(diff, (float, complex)):
            warnings.warn('%s error: %s %s %s' \
                          % (text,y,x,abs(y-x)),  \
                          DebugCudaWarning,stacklevel=2)
        else:
            error_i=np.unravel_index(np.argmax(diff - reltol * abs(y)), \
                                     diff.shape)
            warnings.warn('%s max rel error: %s %s %s %s' \
                          % (text,error_i,y[error_i],x[error_i], \
                             abs(y[error_i]-x[error_i])),  \
                          DebugCudaWarning,stacklevel=2)
            error_i=np.unravel_index(np.argmax(diff),diff.shape)
            warnings.warn('%s max abs error: %s %s %s %s' \
                          % (text,error_i,y[error_i],x[error_i], \
                             abs(y[error_i]-x[error_i])),  \
                          DebugCudaWarning, stacklevel=2)
        
        if raise_error:
            raise DebugCudaError
        return False
    
    return True
