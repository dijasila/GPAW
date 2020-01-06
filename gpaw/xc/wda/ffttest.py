import numpy as np
from ffts import fftn, ifftn

for l in range(10):
    A = np.random.rand(10, 50, 50, 50)
    assert np.allclose(A, ifftn(fftn(A)))
