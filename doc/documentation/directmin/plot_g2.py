# creates: g2.png

import matplotlib.pyplot as plt
import numpy as np

f = plt.figure(figsize=(12, 4), dpi=240)
plt.subplot(121)
# plt.title('Convegrence: ' + r'res$^2$ $< 1\cdot10^{-10} $eV$^2$')

# scf
data = \
['PH3',     18, 
'P2',     14,
'CH3CHO',  17,
'H2COH',   18,
'CS',      15,
'OCHCHO',  17,
'C3H9C',  19,
'CH3COF',  17,
'CH3CH2OCH3', 17,
'HCOOH',   18]
x = data[::2]
y = data[1::2]
plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'b^-', label='SCF', fillstyle='none')

# direct_min
data = \
['PH3', 9,
'P2', 6,
'CH3CHO', 14,
'H2COH', 12,
'CS', 10,
'OCHCHO', 11,
'C3H9C', 13,
'CH3COF', 13,
'CH3CH2OCH3', 12,
'HCOOH', 13]

x = data[::2]
y = np.asarray(data[1::2])+2

plt.plot(range(len(x)), y, 'ro-', label='ETDM',
         fillstyle='none')
plt.legend()
plt.ylabel('Number of iterations (energy and gradients calls)')


plt.subplot(122)

# direct_min
data = \
['NO', 47,
 'CH', 38,
 'OH', 27,
 'ClO', 52,
 'SH', 41]
x = data[::2]
y = np.asarray(data[1::2]) + 2

plt.xticks(range(len(x)), x, rotation=45)
plt.grid(color='k', linestyle=':', linewidth=0.3)
plt.plot(range(len(x)), y, 'ro-', label='ETDM',
         fillstyle='none')
# scf
data = \
['NO', 163,
 'CH', 80,
 'OH', 156,
 'ClO', 117,
 'SH', 35]
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
