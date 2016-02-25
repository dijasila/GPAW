import pickle
import numpy as np

results = pickle.load(open('Si-g0w0.pckl'))
direct_gap = np.min(results['qp'][0,0,-1])-np.max(results['qp'][0,0,-2])

print('Direct bandgap of Si:', direct_gap)
