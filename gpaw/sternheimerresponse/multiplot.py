import numpy as np
import matplotlib.pyplot as plt
import os
import sys

cutoff = int(sys.argv[1])
setup = sys.argv[2]

grid_files = [x for x in os.listdir() if (setup in x and "grid" in x and str(cutoff) in x)]
eigmode_files = [x for x in os.listdir() if (setup in x and "eigmodes" in x and str(cutoff) in x)]
eigval_files = [x for x in os.listdir() if (setup in x and "eigvals" in x and str(cutoff) in x)]

grid = np.load(grid_files[0])
ngrid = np.hstack([-grid[::-1], grid[1:]])
grid = ngrid

eigmodes_SR = np.load(eigmode_files[0])
eigval_S = np.load(eigval_files[0])
print(eigval_S)
pair_S = list(zip(eigval_S, eigmodes_SR))
pair_S = reversed(sorted(pair_S))
eigval_S, eigmodes_SR = zip(*pair_S)
num = 3
eigval_S = eigval_S[:num]
eigmodes_SR = eigmodes_SR[:num]
print(eigval_S)
for S, mode_R in enumerate(eigmodes_SR):
    try:
        plt_mode_R = np.hstack([mode_R[len(mode_R)//2:], mode_R[:len(mode_R)//2]])
        plt_mode_R = plt_mode_R[:-1]
        label = "Eigenvalue: " + str(round(eigval_S[S], 4))
        plt.plot(grid, plt_mode_R, label=label)
    except ValueError as e:
        print("Shape of grid:", grid.shape)
        print("Shape of mode:", mode_R.shape)
        exit()


plt.legend()
plt.title("$\epsilon$ eigenmodes")
plt.xlabel("Distance [A]")
plt.show()
