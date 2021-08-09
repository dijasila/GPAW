# creates: g2.png

import matplotlib.pyplot as plt
import numpy as np


def read_molecules(filename, molnames):
    
    file2read = open(filename, 'r')
    calculated_data_string = file2read.read().split('\n')
    calculated_data = {}
    for i in calculated_data_string:
        if i == '':
            continue
        mol = i.split()
        # ignore last two columns which are memory and elapsed time
        calculated_data[mol[0]] = np.array([float(_) for _ in mol[1:-2]])
    file2read.close()

    data2return = []
    for _ in molnames:
        if 'scf' in filename:
            x = 0
        elif 'dm' in filename:
            x = 1
        data2return.append(_)
        data2return.append(calculated_data[_][x])

    return data2return


f = plt.figure(figsize=(12, 4), dpi=240)
plt.subplot(121)
mollist = \
    ['PH3', 'P2', 'CH3CHO', 'H2COH', 'CS', 'OCHCHO',
     'C3H9C', 'CH3COF', 'CH3CH2OCH3', 'HCOOH']
data = read_molecules('scf-g2-results.txt', mollist)
# scf
x = data[::2]
y = data[1::2]
plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'b^-', label='SCF', fillstyle='none')

# direct_min
data = read_molecules('dm-g2-results.txt', mollist)
x = data[::2]
# add 2 because dm also need 2 diagonalizations
y = np.asarray(data[1::2]) + 2

plt.plot(range(len(x)), y, 'ro-', label='ETDM',
         fillstyle='none')
plt.legend()
plt.ylabel('Number of iterations (energy and gradients calls)')

plt.subplot(122)
# direct_min
mollist = \
    ['NO', 'CH', 'OH', 'ClO', 'SH']
data = read_molecules('dm-g2-results.txt', mollist)
x = data[::2]
y = np.asarray(data[1::2]) + 2

plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'ro-', label='ETDM',
         fillstyle='none')
# scf
data = read_molecules('scf-g2-results.txt', mollist)
x = data[::2]
y = np.asarray(data[1::2])
plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'bo-', label='SCF',
         fillstyle='none')
plt.legend()
plt.ylabel('Number of iterations (energy and gradients calls)')
# plt.text(-6.3, 75.5, '(a)')
# plt.text(-0.7, 75.5, '(b)')
# f.savefig("conv.eps", bbox_inches='tight')
f.savefig("g2.png", bbox_inches='tight')
