import numpy as np

def fftn(n_g):
    N = np.prod(n_g.shape[-3:])

    return np.fft.fftn(n_g, axes=(-3, -2, -1))  / N

def ifftn(n_g):
    N = np.prod(n_g.shape[-3:])

    res = np.fft.ifftn(n_g, axes=(-3, -2, -1)) * N
    assert np.allclose(res, res.real)

    return res.real
    
