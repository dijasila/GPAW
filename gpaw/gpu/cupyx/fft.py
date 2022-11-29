import scipy.fft as fft
import gpaw.gpu.cupy as cp


def ifftn(array_Q,
          shape,
          norm=None,
          overwrite_x=False):
    return cp.ndarray(fft.ifftn(array_Q._data,
                                shape,
                                norm=norm,
                                overwrite_x=overwrite_x))


def fftn(array_Q,
         shape=None,
         norm=None,
         overwrite_x=False):
    return cp.ndarray(fft.fftn(array_Q._data,
                               shape,
                               norm=norm,
                               overwrite_x=overwrite_x))
