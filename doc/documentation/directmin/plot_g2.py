# web-page: g2.png
import matplotlib.pyplot as plt
import numpy as np
from doc.documentation.directmin import tools_and_data


def read_molecules(filename, molnames):

    with open(filename, 'r') as fd:
        calculated_data_string = fd.read()
        calculated_data = \
            tools_and_data.read_data(calculated_data_string)

    data2return = []
    for _ in molnames:
        data2return.append(_)
        data2return.append(calculated_data[_][0])

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
# 2 is added to account for the diagonalization
# performed at the beginning and at the end of etdm
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

f.savefig("g2.png", bbox_inches='tight')
