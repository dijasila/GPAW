import numpy as np

import os

GPAW_CUPY = bool(int(os.environ.get('GPAW_CUPY', 0)))

print('GPAW_CUPY:', GPAW_CUPY)

if GPAW_CUPY:
    from gpaw.gpu.arrays import CuPyArrayInterface
    from gpaw import gpu
    import cupy
    gpu.setup(cuda=True)

def CUPYBridge_inveps(f):
    def identity(*args, **kwargs):
        f(*args, **kwargs)

    # If GPAW_CUPY environment variavble is not set or
    # is set to 0, we will just run the normal function.
    if not GPAW_CUPY:
        return identity

    # If GPAW_CUPY=1, we will convert arrays to cupy-arrays.
    # If they already are cupy arrays, this won't do anything.
    # We do not convert output, and we assign to output, thus
    # cupy will automatically convert to correct output upon
    # assignment (if done properly).
    def bridge(sqrV_g, chi0_GG, mode, fv_GG, weight=1.0, out_GG=None, lib=np):
        if not gpu.is_device_array(out_GG):
            gpu_out_GG = gpu.copy_to_device(out_GG)
        else:
            gpu_out_GG = out_GG
        f(gpu.copy_to_device(sqrV_g),
          gpu.copy_to_device(chi0_GG),
          mode, weight=weight, 
          out_GG=gpu_out_GG, lib=cupy)
        if not gpu.is_device_array(out_GG):
            out_GG[:] = gpu.copy_to_host(gpu_out_GG)
    
    return bridge

@CUPYBridge_inveps
def calculate_inveps(sqrV_G, chi0_GG, mode, fv_GG=None, weight=1.0, out_GG=None, lib=np):
    """

        Calculates the symmetric inverse dielectric function, and adds it to out-array
        with given weight.

        out_GG[:] += weight * inveps_GG

    """
    N = len(sqrV_G)

    if mode == 'GW':
        """
            For GW method, the symmetric dielectric matrix is given as
                  /        ½          ½  \-1
            ε   = | I   - V   chi0   V   |
             GG   \  GG    GG     GG  GG /

        """
        e_GG = -chi0_GG * sqrV_G * sqrV_G[:, lib.newaxis]
        e_GG.flat[::N+1] += 1.0  # Fast way to add an identity matrix
    elif mode == 'GWP':
        raise NotImplementedError
    elif mode == 'GWS':
        raise NotImplementedError
    elif mode == 'GWG':
         raise NotImplementedError
    else:
        raise ValueError(f'Unknown mode: {mode}')

    out_GG += weight * lib.linalg.inv(e_GG)

"""
    def inv(self, a):
        return self.lib.linalg.inv(a)



    def e_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = self.inv(self.I_GG + self._chiVVfv_GG() - self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)


    def e_GG_w_fxc(self):
        return self.I_GG - self._chiVVfv_GG()
"""
