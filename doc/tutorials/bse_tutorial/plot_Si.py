import pylab as pl
import numpy as np

pl.figure()

a = np.loadtxt('eps_rpa_Si.csv', delimiter=',')
pl.plot(a[:, 0], a[:, 4], label='RPA', lw=2)

a = np.loadtxt('eps_bse_Si.csv', delimiter=',')
pl.plot(a[:, 0], a[:, 2], label='BSE', lw=2)

pl.xlabel(r'$\hbar\omega\;[eV]$', size=24)
pl.ylabel(r'$\epsilon_2$', size=24)
pl.xticks(size=20)
pl.yticks(size=20)
pl.tight_layout()
pl.axis([2.0, 6.0, None, None])
pl.legend()

#pl.show()
pl.savefig('bse_Si.png')
