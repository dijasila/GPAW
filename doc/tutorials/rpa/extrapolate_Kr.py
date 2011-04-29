import numpy as np
from pylab import *

A = np.loadtxt('rpa_Kr.dat').transpose()
xs = np.array([170 +i*100. for i in range(500)])

plot(A[1]**(-1.), A[2], 'o', markersize=8, label='Calculated points')
plot(xs**(-1), -10.08+281.705/xs, label='Fit: -10.08+282/bands')
t = [int(A[1,i]) for i in range(len(A[1]))]
t[3] = ''
xticks(A[1]**(-1.), t)
xlabel('bands', fontsize=16)
ylabel('Correlation energy', fontsize=16)
axis([0.,None,None,None])
title('RPA correlation energy of fcc Kr lattice at $V=40\,\AA^3$')
legend(loc='upper left')
savefig('extrapolate_Kr.png')
