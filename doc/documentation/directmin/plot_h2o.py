# creates: water.png
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    
    file2read = open(filename, 'r')
    calculated_data_string = file2read.read().split('\n')
    
    calculated_data = {}
    
    for i in calculated_data_string:
        if i == '':
            continue
        mol = i.split()
        # ignore the last column which are memory
        calculated_data[int(mol[0]) / 3] = \
            np.array([float(_) for _ in mol[1:-1]])

    file2read.close()
    
    return calculated_data


f = plt.figure(figsize=(12, 4), dpi=240)
plt.subplot(121)

# see data from wm_scf.py and wm_dm.py

scf = read_data('scf-water-results.txt')
dm_ui = read_data('dm-water-results.txt')

data2plot = []
for _ in scf.keys():
    data2plot.append(_)
    ratio = scf[_][3] / dm_ui[_][3]
    data2plot.append(ratio)
    assert ratio > 1.3

x = data2plot[::2]
y = data2plot[1::2]

plt.title('Ratio of total elapsed times')
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.ylabel(r'$T_{scf}$ / $T_{etdm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1.0, 3.0)
plt.yticks(np.arange(1, 3.1, 0.5))
plt.plot(x, y, 'bo-')

plt.subplot(122)
# see data from wm_scf.py and wm_dm.py
# add 2 because it also performs diagonalization
# in the begining and the end of etdm

data2plot = []
for _ in scf.keys():
    data2plot.append(_)
    ratio = (scf[_][3] / scf[_][2]) / (dm_ui[_][3] / (dm_ui[_][2] + 2))
    data2plot.append(ratio)
    assert ratio > 1.1

x = data2plot[::2]
y = data2plot[1::2]

plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.title('Ratio of elapsed times per iteration')
plt.ylabel(r'$T_{scf}$ / $T_{etdm}$')
plt.xlabel('Number of water molecules')
plt.ylim(1.0, 3.0)
plt.yticks(np.arange(1, 3.1, 0.5))
plt.plot(x, y, 'ro-')

f.savefig("water.png", bbox_inches='tight')
