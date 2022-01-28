from gpaw.auxlcao.generatedcode import generated_W_LL as generated_W_LL
from gpaw.auxlcao.generatedcode2 import generated_W_LL as generated_W2_LL
import numpy as np

for i in range(10):
    dx = np.random.rand()
    dy = np.random.rand()
    dz = np.random.rand()
    d = (dx**2+dy**2+dz**2)**0.5 

    W_LL = np.zeros((9,9))
    generated_W_LL(W_LL, d, dx, dy, dz)

    W2_LL = np.zeros((9,9))
    generated_W2_LL(2, W2_LL, d, dx, dy, dz)

    print(W_LL)
    print(W2_LL)
    print(W_LL / (W2_LL))
