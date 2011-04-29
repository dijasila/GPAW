import numpy as np
from pylab import *

A = np.loadtxt('rpa_N2.dat').transpose()
plot(A[1]**(-1.), A[2], 'o', label='Calculated points')

xs = np.array([A[1,0]+i*100000. for i in range(50000)])
plot(xs**(-1.), -4.963+958.7*xs**(-1), label='-4.963+959/n')

t = [int(A[1,i]) for i in range(len(A[0]))]
t[-2] = ''
xticks(A[1]**(-1.0), t, fontsize=12)
axis([0.,None, None, -4.])
xlabel('Bands', fontsize=18)
ylabel('RPA correlation energy [eV]', fontsize=18)
legend(loc='lower right')
savefig('extrapolate.png')
