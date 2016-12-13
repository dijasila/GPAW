import pickle
import numpy as np

results = pickle.load(open('Si-g0w0.pckl', 'rb'))
direct_gap = results['qp'][0, 0, -1] - results['qp'][0, 0, -2]

print('Direct bandgap of Si:', direct_gap)
