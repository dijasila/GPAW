import numpy as np
import matplotlib.pyplot as plt
from alphas import get_alphas
 
Z_ig, _ = np.meshgrid(np.linspace(-2.5, 0, 10), np.linspace(0, 1, 10))
Z_ig = Z_ig.T
assert Z_ig.shape == (10, 10)
assert np.allclose(Z_ig[:, 0], Z_ig[:, -2]), f"0: {Z_ig[:, 0]}, -2: {Z_ig[:, -2]}"

alpha_ig = get_alphas(Z_ig)

print(f"Zs: {[round(x, 2) for x in Z_ig[:, 0]]}\nalphas: {alpha_ig[:, 0]}")


runs = 10
for _ in range(runs):
    shape = (10, 10)
    Z_ig = np.zeros(shape)
    for ig in range(shape[1]):
        Zs = (np.random.rand(shape[0]) - 0.5) * 10
        Zs.sort()
        if not (Zs <= -1).any() and (Zs > -1).any():
            Zs[0] = -1
        Zs.sort()
        Z_ig[:, ig] = Zs

    alpha_ig = get_alphas(Z_ig)
    assert np.allclose(alpha_ig.sum(axis=0), 1)
    
