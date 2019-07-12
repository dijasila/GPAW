# creates: solids.png

import glob
import matplotlib.pyplot as plt
import numpy as np

file_pathes = glob.glob('is_*data')
x = {}
y = {}
for i in range(len(file_pathes)):
    name = file_pathes[i][3:-9]
    a = open(file_pathes[i], 'r')
    a = a.read().split('\n')
    x[name] = []
    y[name] = []
    for s in a:
        if s == '':
            continue
        x[name].append(s.split('\t')[0])
        y[name].append(int(s.split('\t')[3]))

types = {'dm': 'ro-',
         'scf': 'b^-',
         'dmuinv': 'go-'}

legends = {'dm': 'Direct Min',
           'scf': 'SCF',
           'dmuinv': 'dm_uinv'}

fillstyle = {'dm': 'none',
             'scf': 'full',
             'dmuinv': 'none'}

f = plt.figure(figsize=(6, 4), dpi=240)
plt.yticks(np.arange(5, 16, step=2))
for i in x.keys():
    if i == 'dmuinv':
        continue
    plt.xticks(range(len(x[i])), x[i], rotation=45)
    plt.plot(range(len(x[i])), y[i], types[i], label=legends[i], fillstyle=fillstyle[i])
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.ylabel('Number of iterations (energy and gradients calls)')
plt.legend()
f.savefig("solids.png", bbox_inches='tight')