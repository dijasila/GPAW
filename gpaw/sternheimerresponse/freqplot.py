import numpy as np
import matplotlib.pyplot as plt
import os
import sys

cutoff = int(sys.argv[1])
setup = sys.argv[2]

def check(fname):
    return (setup in fname and str(cutoff) in fname)

grid_files = [x for x in os.listdir() if ("grid" in x and check(x))]
omegas_files = [x for x in os.listdir() if ("omegas" in x and check(x))]
mode_files = [x for x in os.listdir() if ("eigmodes_wx" in x and check(x))]
eigval_files = [x for x in os.listdir() if ("eigvals_w" in x and check(x))]

grid = np.load(grid_files[0])
ngrid = np.hstack([-grid[::-1], grid[1:]])
grid = ngrid

eigmodes_wR = np.load(mode_files[0])
eigvals_w = np.load(eigval_files[0])
omegas_w = np.load(omegas_files[0])

for w, mode_R in enumerate(eigmodes_wR):
    plt_mode_R = np.hstack([mode_R[len(mode_R)//2:], mode_R[:len(mode_R)//2]])
    plt_mode_R = plt_mode_R[:-1]
    label = "Eigval: {} @ $\omega=${}".format(round(eigvals_w[w], 3), omegas_w[w])
    plt.plot(grid, plt_mode_R, label=label)

plt.legend()
plt.title("$\epsilon$ eigenmodes at different frequencies")
plt.xlabel("Distance [A]")
plt.show()

