import numpy as np
import os
import matplotlib.pyplot as plt
import sys
'''
Load all the eigmode files for the given setup, plot them on top of each, make a legend and save the plot.

Also load all the eigenvalues and make a plot of eigenvalue as function of plane-wave cutoff
'''


setup = sys.argv[1]

grid_files = [x for x in os.listdir() if (setup in x and "grid" in x)]
eigmode_files = [x for x in os.listdir() if (setup in x and "eigmode_" in x)]
eigval_files = [x for x in os.listdir() if (setup in x and "eigval_" in x)]

cuts_to_plot = [100, 200, 400, 600, 900]

grids = []
for fname in grid_files:
    _, cut_str, _, = fname.split("_")
    cut = int(cut_str)
    if not cut in cuts_to_plot:
        continue
    data = np.load(fname)
    grids.append((cut, data))

grids = sorted(grids)

ngrids = []
for cut, grid in grids:
    ngrid = np.hstack([grid, -grid[::-1]])
    ngrids.append((cut, ngrid))

grids = ngrids
    




modes = []
for fname in eigmode_files:
    _, cut_str, _, _ = fname.split("_")
    cut = int(cut_str)
    if not cut in cuts_to_plot:
        continue
    data = np.load(fname)
    modes.append((cut, data))

modes = sorted(modes)
assert len(modes) == len(grids)

cuts1 = np.array([t[0] for t in modes])
cuts2 = np.array([t[0] for t in grids])
assert np.allclose(cuts1, cuts2)

maxVal = np.max(modes[0][1])
for k, (cut, data) in enumerate(modes):
    gcut, gdata = grids[k]
    scale = maxVal/np.max(data)
    plt.plot(grids[k][1], data*scale, label=str(cut))
    
                   
plt.title(setup + " eigenmodes", fontsize=20)  
plt.legend()
plt.savefig(setup + "modes.png")
    
    
cuts_to_plot = [int(x) for x in np.arange(100, 1000, 50)]
cut_eigval = []
for fname in eigval_files:
    _, cut_str, _ = fname.split("_")
    cut = int(cut_str)
    if not cut in cuts_to_plot:
        continue
    data = np.load(fname)
    data = float(data.real)
    cut_eigval.append((cut, data))

cut_eigval = sorted(cut_eigval)
    

cuts = [t[0] for t in cut_eigval]


vals = [t[1] for t in cut_eigval]

plt.figure()
plt.plot(cuts, vals, marker='.')
plt.title(setup + " eigenvals", fontsize=20)
plt.savefig(setup + "eigenvals.png")






