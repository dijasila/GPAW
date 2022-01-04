"""Plot magnon energy as a function of rc for all
high-symmetry points of Co(hcp)"""

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import json

# ----- Load results ----- #

# From 'high_sym_pts.py'
rc_r = np.load('rc_r.npy')
E_rmq = np.load('high_sym_pts_E_rmq.npy')
with open('spts.json') as file:
    spts = json.load(file)

# Get info
Nr, N_sites, Nq = E_rmq.shape
qnames = list(spts.keys())